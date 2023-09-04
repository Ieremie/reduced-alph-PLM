import torch.nn as nn

class ProSEMT(nn.Module):
    def __init__(self, embedding):
        super(ProSEMT, self).__init__()
        self.skipLSTM = embedding
    def forward(self, seq_unpacked, lens_unpacked, apply_proj=True):
        return self.skipLSTM(seq_unpacked, lens_unpacked, apply_proj)