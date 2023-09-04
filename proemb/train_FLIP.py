import os
from argparse import ArgumentParser
from pathlib import Path
import random

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from proemb import alphabets
from proemb.datasets import FlipDataset
from proemb.models.protein_sequence_heads import SequenceRegression
from proemb.models.lstm import SkipLSTM
from proemb.utils.util import collate_seq_regression

from pytorch_lightning import loggers as pl_loggers

import pprint


def main(args):
    # ----------------------------------------------EXPERIMENT SETUP-------------------------------------------
    if args.remote:
        experiment_id = f"flip/{str(os.environ['SLURM_JOB_NAME'])}/{str(os.environ['SLURM_JOB_ID'])}"

        Path(f"/scratch/ii1g17/protein-embeddings/runs/{experiment_id}/lightning_logs").mkdir(parents=True, exist_ok=True)
        # make dir if it doesn't exist
        save_dir = Path(f"/scratch/ii1g17/protein-embeddings/saved_models/{experiment_id}/")
        save_dir.mkdir(parents=True, exist_ok=True)

    else:
        experiment_id = 'local'
        save_dir = None

    pprint.pprint(vars(args))
    # ----------------------------------------------DATASETS------------------------------------------------------------
    alphabet = alphabets.get_alphabet(args.alphabet_type)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch_generator = torch.Generator().manual_seed(args.seed)

    train_dataset = FlipDataset(base_pth=args.data_path, dataset=args.dataset, split=args.split,
                                type='train', augment_prob=args.augment_prob, noise_type=args.noise_type,
                                alphabet=alphabet, full_train=args.random_validation)

    # validation is 10% of the training set
    if args.random_validation:
        val_s = int(args.val_split * len(train_dataset))
        train_s = len(train_dataset) - val_s
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_s, val_s],
                                                                   generator=torch_generator)
    else:
        val_dataset = FlipDataset(base_pth=args.data_path, dataset=args.dataset, split=args.split,
                                  type='val', augment_prob=args.augment_prob, noise_type=args.noise_type,
                                  alphabet=alphabet)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_seq_regression,
                              num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_seq_regression, num_workers=8)

    test_loader = DataLoader(
        FlipDataset(base_pth=args.data_path, dataset=args.dataset, split=args.split,
                    type='test', augment_prob=args.augment_prob, noise_type=args.noise_type, alphabet=alphabet),
        batch_size=args.batch_size, collate_fn=collate_seq_regression, num_workers=8)

    # print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_loader)}")

    # ----------------------------------------------MODEL------------------------------------------------------------
    encoder = SkipLSTM(21, 100, 512, 3)
    # only using the encoder weights
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    encoder_checkpoint = {k.replace('skipLSTM.', ''): v for k, v in checkpoint['model'].items() if
                          k.startswith('skipLSTM')}
    encoder.load_state_dict(encoder_checkpoint, strict=False)

    if args.proj_head is None:
        embedding_dim = encoder.proj.in_features
    if args.proj_head == 'sim':
        embedding_dim = encoder.proj.out_features
    if args.proj_head == 'cloze':
        embedding_dim = encoder.cloze.out_features

    model = SequenceRegression(encoder=encoder, emb_dim=embedding_dim, head_type=args.head_type,
                               hidden_dim=args.hidden_dim, dropout_head=args.dropout_head,
                               proj_head=args.proj_head, unfreeze_encoder_epoch=args.unfreeze_encoder_epoch,
                               encoder_lr=args.encoder_lr, learning_rate=args.lr,
                               factor=args.factor, scheduler_patience=args.scheduler_patience,
                               num_classes=1, task='FLIP')

    early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True, dirpath=save_dir)

    # ----------------------------------------------TRAINING------------------------------------------------------------
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"/scratch/ii1g17/protein-embeddings/runs/{experiment_id}")
    trainer = Trainer.from_argparse_args(args, logger=tb_logger, accelerator='gpu' if args.remote else 'cpu', devices=1,
                                         callbacks=[early_stopping, checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)

    # ----------------------------------------------TESTING------------------------------------------------------------
    trainer.test(dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path,
                 verbose=True)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str, default='../data/FLIP', help='Path to the data')
    parser.add_argument('--dataset', type=str, default='aav', help='dataset name (default: aav)')
    parser.add_argument('--split', type=str, default='des_mut', help='FLIP dataset split')
    parser.add_argument('--model_path', type=str, help='path to LSTM embedding model',
                        default='../data/saved-models/iter_240000_checkpoint.pt')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden units of MLP (default: 1024)')

    parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 256)')
    parser.add_argument('--max_epochs', type=int, default=10000, help='number of epochs (default: 100)')
    parser.add_argument('--gradient_clip_val', type=float, default=3, help='gradient clipping (default: 3.0)')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--encoder_lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout_head', type=float, default=0, help='dropout probability for MLP (default: 0.5)')
    parser.add_argument('--unfreeze_encoder_epoch', type=int
                        , default=10, help='unfreeze encoder after this many epochs (default: 2000 never)')
    parser.add_argument('--remote', action='store_true', help='run on iridis')
    parser.add_argument('--alphabet_type', type=str, default='uniprot21', help='alphabet type (default: uniprot21)')

    parser.add_argument('--augment_prob', type=float, default=0, help='augment probability (default: 0.0)')
    parser.add_argument('--head_type', type=str, default='mlp3', help='head type (default: mlp)')
    parser.add_argument('--proj_head', type=str, default=None, help='projection head (e.g cloze, sim)')
    parser.add_argument('--noise_type', type=str, default='pfam-hmm', help='noise type (default: pfam-hmm)')

    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping (default: 20)')
    parser.add_argument('--factor', type=float, default=0.6, help='factor for lr scheduler (default: 0.6)')
    parser.add_argument('--scheduler_patience', type=int, default=3, help='patience for lr scheduler (default: 3)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument('--random_validation', action='store_true', help='use random validation set')
    parser.add_argument('--val_split', type=float, default=0.1, help='validation split (default: 0.1)')

    args = parser.parse_args()

    main(args)
