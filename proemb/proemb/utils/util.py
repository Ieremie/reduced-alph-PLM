from typing import Tuple
import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_sequence, pad_sequence



class LargeWeightedRandomSampler(torch.utils.data.sampler.WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)), size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


def collate_protein_seq(args):
    """
    Sorting seq and labels in a batch in decreasing order
    Args:
       seqs ((torch.FloatTensor, ...) tuple): protein sequences in a batch
       labels ((torch.FloatTensor, ...) tuple): labels of the original masked AAs in a batch
       (same length as the protein seqs)
    Returns:
        torch.FloatTensor: containing the padded sequence, shape: (Batch * MaxL)
        torch.FloatTensor: containing the lengts of the sequences in the batch, shape: (Batch, )
        PackedSequence: contains the labels of the sequences in the batch (original values of masked AAs)
    """

    seqs, labels = zip(*args)
    # from torch tensor to normal list
    seqs = list(seqs)
    labels = list(labels)

    # sort them by decreasing length to work with pytorch ONNX
    seqs = sorted(seqs, key=lambda x: x.size()[0], reverse=True)
    labels = sorted(labels, key=lambda x: x.size()[0], reverse=True)

    seqs_lengths = torch.FloatTensor([len(seq) for seq in seqs])
    # using pad_sequence instead of directly using ge due to multi GPU training
    # padding value same with unknown amino acid A=20, this does not affect training as we are packing the seqs later
    seqs = pad_sequence(seqs, batch_first=True, padding_value=20)
    labels = pack_sequence(labels, enforce_sorted=False)

    return seqs, seqs_lengths, labels


def collate_seq_classification(data: Tuple):
    seqs, labels, names, fasta = zip(*data)
    seqs_lengths = torch.FloatTensor([len(seq) for seq in seqs])
    seqs = pad_sequence(seqs, batch_first=True, padding_value=20)

    return seqs.to(torch.int64), seqs_lengths.to(torch.int64), torch.LongTensor(labels)


def collate_seq_regression(data: Tuple):
    seqs, targets = zip(*data)
    seqs_lengths = torch.FloatTensor([len(seq) for seq in seqs])
    seqs = pad_sequence(seqs, batch_first=True, padding_value=20)

    return seqs.to(torch.int64), seqs_lengths.to(torch.int64), torch.FloatTensor(targets)
