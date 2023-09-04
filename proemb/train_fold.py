import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from proemb import alphabets
from proemb.datasets import FoldHomologyDataset
from proemb.models.protein_sequence_heads import SequenceClassifier
from proemb.models.lstm import SkipLSTM
from proemb.utils.util import collate_seq_classification

from pytorch_lightning import loggers as pl_loggers

import pprint


def main(args):
    # ----------------------------------------------EXPERIMENT SETUP-------------------------------------------
    if args.remote:
        experiment_id = f"fold/{str(os.environ['SLURM_JOB_NAME'])}/{str(os.environ['SLURM_JOB_ID'])}"
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

    train_loader = DataLoader(
        FoldHomologyDataset(dataset_pth=args.fold_data_path, data_split='training', alphabet=alphabet,
                            augment_prob=args.augment_prob, noise_type=args.noise_type),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_seq_classification, num_workers=8)

    val_loader = DataLoader(
        FoldHomologyDataset(dataset_pth=args.fold_data_path, data_split='validation', alphabet=alphabet,
                            noise_type=None),
        batch_size=args.batch_size, collate_fn=collate_seq_classification, num_workers=8)

    test_fold_loader = DataLoader(
        FoldHomologyDataset(dataset_pth=args.fold_data_path, data_split='test_fold', alphabet=alphabet,
                            noise_type=None),
        batch_size=args.batch_size, collate_fn=collate_seq_classification, num_workers=8)

    test_superfamily_loader = DataLoader(
        FoldHomologyDataset(dataset_pth=args.fold_data_path, data_split='test_superfamily', alphabet=alphabet,
                            noise_type=None),
        batch_size=args.batch_size, collate_fn=collate_seq_classification, num_workers=8)

    test_family_loader = DataLoader(
        FoldHomologyDataset(dataset_pth=args.fold_data_path, data_split='test_family', alphabet=alphabet,
                            noise_type=None),
        batch_size=args.batch_size, collate_fn=collate_seq_classification, num_workers=8)

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

    model = SequenceClassifier(encoder=encoder, emb_dim=embedding_dim, head_type=args.head_type,
                               hidden_dim=args.hidden_dim, dropout_head=args.dropout_head,
                               proj_head=args.proj_head, unfreeze_encoder_epoch=args.unfreeze_encoder_epoch,
                               encoder_lr=args.encoder_lr, learning_rate=args.lr,
                               factor=args.factor, scheduler_patience=args.scheduler_patience,
                               task='FOLD', num_classes=1195)

    mode = 'min' if args.early_stopping_metric == 'val_loss' else 'max'
    early_stopping = EarlyStopping(monitor=args.early_stopping_metric, patience=args.patience, verbose=True, mode=mode)
    checkpoint_callback = ModelCheckpoint(monitor=args.early_stopping_metric, mode=mode, save_top_k=1, verbose=True,
                                          dirpath=save_dir)
    # ----------------------------------------------TRAINING------------------------------------------------------------
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"/scratch/ii1g17/protein-embeddings/runs/{experiment_id}")
    trainer = Trainer.from_argparse_args(args, logger=tb_logger, accelerator='gpu' if args.remote else 'cpu', devices=1,
                                         callbacks=[early_stopping, checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)

    # ----------------------------------------------TESTING------------------------------------------------------------
    trainer.test(dataloaders=[test_fold_loader, test_superfamily_loader, test_family_loader],
                 ckpt_path=checkpoint_callback.best_model_path, verbose=True)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--fold_data_path', type=str, default='../data/HomologyTAPE', help='Path to the data')
    parser.add_argument('--model_path', type=str, help='path to LSTM embedding model',
                        default='../data/saved-models/iter_240000_checkpoint.pt')
    parser.add_argument('--hidden_dim', type=int
                        , default=1024, help='hidden units of MLP (default: 1024)')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size (default: 64)')
    parser.add_argument('--max_epochs', type=int, default=100, help='number of epochs (default: 100)')
    parser.add_argument('--gradient_clip_val', type=float, default=3, help='gradient clipping (default: 3.0)')

    parser.add_argument('--lr', type=float, default=0.00017, help='learning rate (default: 0.001)')
    parser.add_argument('--encoder_lr', type=float, default=0.00005, help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout_head', type=float, default=0.7, help='dropout probability for MLP (default: 0.5)')
    parser.add_argument('--unfreeze_encoder_epoch', type=int
                        , default=35, help='unfreeze encoder after this many epochs (default: 2000 never)')
    parser.add_argument('--remote', action='store_true', help='run on iridis cluster')
    parser.add_argument('--alphabet_type', type=str, default='uniprot21', help='alphabet type (default: uniprot21)')

    parser.add_argument('--augment_prob', type=float, default=0, help='augment probability (default: 0.0)')
    parser.add_argument('--head_type', type=str, default='mlp', help='head type (default: mlp)')
    parser.add_argument('--proj_head', type=str, default=None, help='projection head (e.g cloze, sim)')
    parser.add_argument('--noise_type', type=str, default='pfam-hmm', help='noise type (default: pfam-hmm)')

    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping (default: 20)')
    parser.add_argument('--factor', type=float, default=0.6, help='factor for lr scheduler (default: 0.6)')
    parser.add_argument('--scheduler_patience', type=int, default=3, help='patience for lr scheduler (default: 3)')
    parser.add_argument('--early_stopping_metric', type=str, default='val_loss',
                        help='early stopping metric (default: val_loss)')

    args = parser.parse_args()

    main(args)
