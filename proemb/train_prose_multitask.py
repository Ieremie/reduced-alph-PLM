from __future__ import print_function, division

import argparse
import os
import pprint
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from catalyst.data import DistributedSamplerWrapper
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

from proemb import alphabets
from proemb.datasets import ClozeDataset, LMDBDataset
from proemb.models.lstm import SkipLSTM
from proemb.models.language_model import ProSEMT
from proemb.utils.train_utils import multi_gpu_info, itr_restart, \
    epoch_time, multitask_train_info, save_checkpoint
from proemb.utils.util import LargeWeightedRandomSampler, collate_protein_seq

torch_generator = torch.Generator().manual_seed(42)


def cloze_grad(model, seqs, seqs_lengths, labels, device, weight=1.0, backward=True, use_multi_gpu=True):
    seqs = seqs.to(device)
    labels = labels.data.to(device)

    # we only consider AA that have been masked (this also exclude the AA that have the unknwon token '20')
    mask = (labels < 20)
    # check that we have noised positions...
    loss = 0
    correct = 0
    n = mask.float().sum().item()
    if n > 0:
        padded_output = model(seqs, seqs_lengths, apply_proj=False)
        logits = pack_padded_sequence(padded_output, seqs_lengths.to('cpu'), batch_first=True,
                                      enforce_sorted=False)

        z = logits.data
        # using the original projection layer to project into 21 dim
        model = model.module.skipLSTM if use_multi_gpu else model.skipLSTM
        logits = model.cloze(z)
        logits = logits[mask]
        labels = labels[mask]

        loss = F.cross_entropy(logits, labels)
        _, y_hat = torch.max(logits, 1)

        w_loss = loss * weight

        inputs = [p for p in list(model.layers.parameters()) + list(model.cloze.parameters()) if p.requires_grad]
        if backward and inputs:
            w_loss.backward(inputs=inputs)

        loss = loss.item()
        correct = torch.sum((labels == y_hat).float()).item()

    return loss, correct, n


def eval_cloze(model, data_it, device, use_multi_gpu=True):
    losses, accuracies = [], []
    for seqs, seqs_lengths, labels in data_it:
        loss, correct, nr_masked = cloze_grad(model, seqs, seqs_lengths, labels, device, weight=1.0, backward=False,
                                              use_multi_gpu=use_multi_gpu)

        if nr_masked > 0:
            losses.append(loss)
            accuracies.append(correct / nr_masked)
    return np.mean(losses), np.mean(accuracies)


def average_gradients(dist, model, world_size):
    """Calculates the average of the gradients over all gpus.
    """
    for param in model.parameters():
        if param.grad is not None:
            dist.barrier()
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
    return True


def freeze_unfreeze_params(params, freeze=True):
    for param in params:
        param.requires_grad = not freeze


# ---------------------------------API PARSING-------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser('Script for training multitask embedding model')

    data_path = '/scratch/ii1g17/protein-embeddings/'

    parser.add_argument('--path-train', default=f'{data_path}/uniref-2018july/uniref90/uniref90.fasta.lmdb')
    parser.add_argument('--base', default=f'{data_path}data/')

    parser.add_argument('--rnn-dim', type=int, default=512, help='hidden units of RNNs (default: 512)')
    parser.add_argument('--num-layers', type=int, default=3, help='number of RNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0, help='dropout probability (default: 0)')
    parser.add_argument('--embedding-dim', type=int, default=100, help='embedding dimension (default: 100)')

    parser.add_argument('-n', '--num-steps', type=int, default=2000000, help='number ot training steps (default:1 mil)')
    parser.add_argument('--max-length', type=int, default=500, help='sample seq down to (500) during training')
    parser.add_argument('-p', type=float, default=0.1, help='cloze residue masking rate (default: 0.1)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-4)')

    parser.add_argument('--log-freq', type=int, default=10, help='logging information every e.g. 100 batches')
    parser.add_argument('-o', '--output', help='output file path (default: stdout)')
    parser.add_argument('--use-multi-gpu', action='store_true', help='using multiple gpus in parallel for training')

    parser.add_argument('--cloze-weight', type=float, default=1)
    parser.add_argument('--cloze', action='store_true', help='grad for cloze head')

    parser.add_argument('--cloze-batch-size', type=int, default=64,
                        help='minibatch size for the cloze loss (default: 64)')

    parser.add_argument('--augment', type=float, default=0.05,
                        help='resample amino acids during training with this probability (default: 0.05)')

    parser.add_argument('--hack-offset', type=int, default=0, help='hack to use 4 extra rtx8000')
    parser.add_argument('--save-interval', type=int, default=20000, help='frequency of saving (default:; 100,000)')

    parser.add_argument("--model-checkpoint", type=str, default=None, help="path to model to load")
    parser.add_argument("--pretrained-encoder", type=str, default=None, help="path to encoder to load")

    parser.add_argument('--alphabet-type', type=str, default='uniprot21', help='alphabet to use (default: uniprot21)')
    parser.add_argument('--backprop-lm-step', type=int, default=5000, help='backprop lm step (default: 100000)')
    parser.add_argument('--increased-model-size', action='store_true', help='ignore projection loading')

    args = parser.parse_args()
    # ---------------------------------DISTRIBUTED PARALLEL TRAINING---------------------------------------------------

    output = sys.stdout if args.output is None else open(args.output, 'w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_multi_gpu = args.use_multi_gpu

    rank = 0
    experiment_id = "single-gpu"
    if use_multi_gpu:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["SLURM_PROCID"]) + args.hack_offset
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)

        multi_gpu_info(rank, world_size, gpus_per_node, local_rank)
        device = local_rank
        output = open(os.devnull, 'w') if rank != 0 else output

        # we divide by nr_gpus because each GPU gets a section of the data
        args.num_steps = args.num_steps // world_size

        experiment_id = f"multitask/{str(os.environ['SLURM_JOB_NAME'])}/{str(os.environ['SLURM_JOB_ID'])}"

    writer = SummaryWriter(log_dir=f"/scratch/ii1g17/protein-embeddings/runs/{experiment_id}")
    pprint.pprint(args.__dict__, width=1, stream=output)

    # ----------------------------------------DATASETS and DATA LOADERS------------------------------------------------
    alphabet = alphabets.get_alphabet(args.alphabet_type)

    # ----------------------------- CLOZE---------------------------------------------
    if args.cloze:
        fasta_train = LMDBDataset(args.path_train, max_length=args.max_length,
                                  alphabet=alphabet, output=output)
        cloze_train = ClozeDataset(fasta_train, args.p, noise=fasta_train.noise)

        val_size = int(0.1 / 100 * len(cloze_train))
        train_size = len(cloze_train) - val_size
        cloze_train, cloze_validation = random_split(cloze_train, [train_size, val_size], generator=torch_generator)
        # the split for the lengths is the same as the one in the dataset
        cloze_train_lengths, _ = random_split(fasta_train.lengths, [train_size, val_size], generator=torch_generator)

        weight = np.maximum(np.array(cloze_train_lengths) / args.max_length, 1)
        sampler = LargeWeightedRandomSampler(weight, args.cloze_batch_size * args.num_steps)

        sampler = DistributedSamplerWrapper(sampler, num_replicas=world_size, rank=rank) if use_multi_gpu else sampler
        cloze_iterator = itr_restart(DataLoader(cloze_train, batch_size=args.cloze_batch_size, sampler=sampler,
                                                collate_fn=collate_protein_seq, num_workers=2))

        if rank == 0:
            cloze_val_it = DataLoader(cloze_validation, batch_size=args.cloze_batch_size,
                                      collate_fn=collate_protein_seq, num_workers=2)

    # ----------------------------------------MODEL CREATION----------------------------------------------------------
    encoder = SkipLSTM(21, args.embedding_dim, args.rnn_dim, args.num_layers, dropout=args.dropout)
    model = ProSEMT(encoder)

    step = 0
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    # linearly decayed to one tenth of its peak value over the 90% of training duration.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: 1 - s / args.num_steps * 0.9)
    print('Model size: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6), file=output)

    # ----------------------------------------LOAD CHECKPOINT----------------------------------------------------------

    map_location = {'cuda:0': 'cuda:%d' % local_rank} if use_multi_gpu else None
    if args.model_checkpoint is not None:
        print("--------USING CHECKPOINT MODEL-----------", file=output)
        checkpoint = torch.load(args.model_checkpoint, map_location=map_location)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimiser.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        step = checkpoint['step']

    frozen_params = []
    if args.pretrained_encoder is not None:
        print("--------USING PRETRAINED ENCODER-----------", file=output)
        checkpoint = torch.load(args.pretrained_encoder, map_location=map_location)
        # keep only the encoder
        encoder_checkpoint = {k: v for k, v in checkpoint['model'].items() if k.startswith('skipLSTM')
                              and not k.startswith('skipLSTM.proj')}

        # proj head needs to be reinitialised as it is not the same size
        if args.increased_model_size:
            step = checkpoint['step']
            encoder_checkpoint = {k: v for k, v in encoder_checkpoint.items() if not k.startswith('skipLSTM.cloze')}

        model.load_state_dict(encoder_checkpoint, strict=False)
        # freeze the loaded bit until the unfreeze step is reached
        frozen_params = [param for name, param in model.named_parameters() if name in encoder_checkpoint]
        freeze_unfreeze_params(frozen_params, freeze=True)

    # ----------------------------------------DISTRIBUTED----------------------------------------------------------
    if use_multi_gpu:
        model = DDP(model, device_ids=[local_rank])
    model.train()

    # ----------------------------------------------TRAINING-----------------------------------------------------------

    start_time = time.time()
    for i in range(step, args.num_steps):

        if i - step == args.backprop_lm_step and args.pretrained_encoder is not None:
            print('Unfreezing encoder layers', file=output)
            freeze_unfreeze_params(frozen_params, freeze=False)

        with model.no_sync() if use_multi_gpu else nullcontext() as gs:
            # ----------------------------------------------LANGUAGE MODEL---------------------------------------------
            if args.cloze:
                seqs, seqs_lengths, labels = next(cloze_iterator)
                cloze_loss, correct, nr_masked = cloze_grad(model, seqs, seqs_lengths, labels, device,
                                                            weight=args.cloze_weight, use_multi_gpu=args.use_multi_gpu)
                if nr_masked > 0:
                    cloze_acc = correct / nr_masked

        # ----------------------------------------------PARAMETER UPDATES----------------------------------------------

        '''gradients are not syncronized because of multitask and reentrant backwards, so no_syc() was used
           we have to syncronize them manually (no time wasted)
        this also makes processing wait when testing is done on 1 process because they have reached a barrier
        but the one doing the testing (rank 0) has not encountered the barrier yet'''
        if args.use_multi_gpu:
            average_gradients(dist, model, world_size)

        optimiser.step()
        scheduler.step()
        optimiser.zero_grad()
        # ----------------------------------------------TRAINING UPDATES----------------------------------------------
        if i % args.log_freq == 0 and rank == 0:
            epoch_mins, epoch_secs = epoch_time(start_time, time.time())
            start_time = time.time()
            print(f'{i}/{args.num_steps} training {i / args.num_steps:.1%},\
             {epoch_mins}m {epoch_secs}s', file=output)

            if args.cloze and nr_masked > 0:
                multitask_train_info(writer, np.exp(cloze_loss), cloze_acc, i, data="train", name="LM")

        # evaluate and save model
        if i % args.save_interval == 0 and i != 0 and rank == 0:
            # ----------------------------------------------TESTING----------------------------------------------------
            # eval and save model
            model.eval()
            with torch.no_grad():
                if args.cloze:
                    cloze_val_loss, cloze_val_acc = eval_cloze(model, cloze_val_it, device)
                    multitask_train_info(writer, np.exp(cloze_val_loss), cloze_val_acc, i, data="val", name="LM")

            # ----------------------------------------------SAVING MODEL-------------------------------------------
            save_checkpoint(model, optimiser, i, experiment_id, use_multi_gpu, scheduler)

            # flip back to train mode
            model.train()

    if rank == 0:
        save_checkpoint(model, optimiser, args.num_steps, experiment_id, use_multi_gpu, scheduler)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
