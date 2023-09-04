import torch
import pytorch_lightning as pl
from scipy.stats import pearsonr, stats
from torch import nn
from torch.optim import lr_scheduler


class AttentionPooling(nn.Module):
    def __init__(self, input_size):
        super(AttentionPooling, self).__init__()
        self.score = nn.Linear(input_size, 1)

    def forward(self, inputs):
        scores = self.score(inputs)
        weights = torch.softmax(scores, dim=1)
        weighted_inputs = inputs * weights
        output = torch.sum(weighted_inputs, dim=1)
        return output


class SimpleMLP(nn.Module):
    """
    Apply pooling to the input and then apply MLP
    """

    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(SimpleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        self.pooling = AttentionPooling(input_size)

    def forward(self, inputs, seq_lengths):
        pooled = self.pooling(inputs)
        output = self.mlp(pooled)
        return output


class SimpleMLP2(nn.Module):
    """
    Apply MLP to each token and then pool the output
    Each AA predicts the EC number
    """

    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(SimpleMLP2, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        self.pooling = AttentionPooling(output_size)

    def forward(self, inputs, seq_lengths):
        output = self.mlp(inputs)
        pooled = self.pooling(output)
        return pooled


class SimpleMLP3(nn.Module):
    """
    Apply MLP to each token and then pool the output
    """

    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(SimpleMLP3, self).__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        self.pooling = AttentionPooling(hidden_size)

    def forward(self, inputs, seq_lengths):
        projected = self.proj(inputs)
        pooled = self.pooling(projected)
        output = self.mlp(pooled)
        return output



class SequenceClassifier(pl.LightningModule):
    def __init__(self, encoder, emb_dim, head_type, hidden_dim, dropout_head,
                 proj_head, unfreeze_encoder_epoch, encoder_lr, learning_rate, factor, scheduler_patience,
                 num_classes=384, task='Enzyme'):
        super().__init__()
        self.freezed = None
        self.encoder = encoder
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.head_type = head_type
        self.dropout_head = dropout_head
        self.proj_head = proj_head
        self.unfreeze_encoder_epoch = unfreeze_encoder_epoch
        self.encoder_lr = encoder_lr
        self.learning_rate = learning_rate
        self.factor = factor
        self.scheduler_patience = scheduler_patience
        self.num_classes = num_classes
        self.task = task

        if self.head_type == 'mlp':
            self.head = SimpleMLP(self.emb_dim, self.hidden_dim, self.num_classes, self.dropout_head)
        elif self.head_type == 'mlp2':
            self.head = SimpleMLP2(self.emb_dim, self.hidden_dim, self.num_classes, self.dropout_head)
        elif self.head_type == 'mlp3':
            self.head = SimpleMLP3(self.emb_dim, self.hidden_dim, self.num_classes, self.dropout_head)

        self.freeze_encoder()

    def freeze_encoder(self):
        print('-------Freezing encoder---------')
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freezed = True

    def unfreeze_encoder(self):
        print('-------Unfreezing encoder---------')
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freezed = False

    def forward(self, seqs, seq_lengths):

        seq_lengths = seq_lengths.to('cpu')
        apply_proj = True if self.proj_head == 'sim' else False
        lm_output = self.encoder(seqs.to(self.device), seq_lengths, apply_proj=apply_proj)
        if self.proj_head == 'cloze':
            lm_output = self.encoder.cloze(lm_output)
        return self.head(lm_output, seq_lengths)

    def _shared_eval_step(self, batch, batch_idx):
        seqs, seqs_lengths, labels = batch
        logits = self(seqs, seqs_lengths)

        loss = nn.CrossEntropyLoss()(logits, labels)
        acc = torch.mean((torch.argmax(logits, dim=1) == labels).float())
        return loss, acc

    def training_step(self, batch, batch_idx):

        if self.current_epoch == self.unfreeze_encoder_epoch and self.freezed:
            self.unfreeze_encoder()
            self.trainer.optimizers[0].add_param_group({'params': self.encoder.parameters(),
                                                        'lr': self.encoder_lr})
            self.lr_schedulers().min_lrs.append(0.000001)

        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log(f'{self.task}/loss', {"train": loss}, on_step=False, on_epoch=True)
        self.log(f'{self.task}/accuracy', {"train": acc}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log(f'{self.task}/loss', {"val": loss}, on_step=False, on_epoch=True)
        self.log(f'{self.task}/accuracy', {"val": acc}, on_step=False, on_epoch=True)
        # used for scheduler and early stopping
        self.log('val_loss', loss, logger=False)
        self.log('val_acc', acc, logger=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log(f'{self.task}/loss', {f"test": loss}, on_step=False, on_epoch=True)
        self.log(f'{self.task}/accuracy', {f"test": acc}, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.head.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=self.factor,
                                                   patience=self.scheduler_patience, verbose=True,
                                                   min_lr=[0.000001])
        return [opt], [{'scheduler': scheduler, 'monitor': 'val_loss'}]


class SequenceRegression(pl.LightningModule):
    def __init__(self, encoder, emb_dim, head_type, hidden_dim, dropout_head,
                 proj_head, unfreeze_encoder_epoch, encoder_lr, learning_rate, factor, scheduler_patience,
                 num_classes=1, task='FLIP'):
        super().__init__()
        self.freezed = None
        self.encoder = encoder
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.head_type = head_type
        self.dropout_head = dropout_head
        self.proj_head = proj_head
        self.unfreeze_encoder_epoch = unfreeze_encoder_epoch
        self.encoder_lr = encoder_lr
        self.learning_rate = learning_rate
        self.factor = factor
        self.scheduler_patience = scheduler_patience
        self.num_classes = num_classes
        self.task = task

        if self.head_type == 'mlp':
            self.head = SimpleMLP(self.emb_dim, self.hidden_dim, self.num_classes, self.dropout_head)
        elif self.head_type == 'mlp2':
            self.head = SimpleMLP2(self.emb_dim, self.hidden_dim, self.num_classes, self.dropout_head)
        elif self.head_type == 'mlp3':
            self.head = SimpleMLP3(self.emb_dim, self.hidden_dim, self.num_classes, self.dropout_head)

        self.freeze_encoder()

    def freeze_encoder(self):
        print('-------Freezing encoder---------')
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freezed = True

    def unfreeze_encoder(self):
        print('-------Unfreezing encoder---------')
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freezed = False

    def forward(self, seqs, seq_lengths):

        seq_lengths = seq_lengths.to('cpu')
        apply_proj = True if self.proj_head == 'sim' else False
        lm_output = self.encoder(seqs.to(self.device), seq_lengths, apply_proj=apply_proj)
        if self.proj_head == 'cloze':
            lm_output = self.encoder.cloze(lm_output)
        return self.head(lm_output, seq_lengths)

    def _shared_eval_step(self, batch, batch_idx):
        seqs, seqs_lengths, labels = batch
        logits = self(seqs, seqs_lengths)

        # [Batch, 1] -> [Batch]
        logits = logits.squeeze(1)

        loss = nn.MSELoss()(logits, labels)
        rho, _ = stats.spearmanr(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())
        return loss, rho

    def training_step(self, batch, batch_idx):

        if self.current_epoch == self.unfreeze_encoder_epoch and self.freezed:
            self.unfreeze_encoder()
            self.trainer.optimizers[0].add_param_group({'params': self.encoder.parameters(),
                                                        'lr': self.encoder_lr})
            self.lr_schedulers().min_lrs.append(0.000001)

        loss, corr = self._shared_eval_step(batch, batch_idx)
        self.log(f'{self.task}/loss', {"train": loss}, on_step=False, on_epoch=True)
        self.log(f'{self.task}/corr', {"train": corr}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, corr = self._shared_eval_step(batch, batch_idx)
        self.log(f'{self.task}/loss', {"val": loss}, on_step=False, on_epoch=True)
        self.log(f'{self.task}/corr', {"val": corr}, on_step=False, on_epoch=True)
        # used for scheduler and early stopping
        self.log('val_loss', loss, logger=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        loss, corr = self._shared_eval_step(batch, batch_idx)
        self.log(f'{self.task}/loss', {f"test": loss}, on_step=False, on_epoch=True)
        self.log(f'{self.task}/corr', {f"test": corr}, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.head.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=self.factor,
                                                   patience=self.scheduler_patience, verbose=True,
                                                   min_lr=[0.000001])
        return [opt], [{'scheduler': scheduler, 'monitor': 'val_loss'}]
