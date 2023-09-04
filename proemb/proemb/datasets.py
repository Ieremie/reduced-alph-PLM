""" Adapted from https://github.com/tbepler/prose and
     https://github.com/songlab-cal/tape """
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path

import lmdb
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from torch.utils.data.dataset import Dataset

from proemb.alphabets import PFAM_INDEX_TO_AA
from proemb.alphabets import Uniprot21
from proemb.utils.enzyme_util import PDB_OBSOLETE_REMAP, MIXED_SPLIT_CHAINS
from proemb.utils.hmm_util import setup_hmm_dict, get_seq_hmm_probabilities


def uniref90_agument(noise, length):
    return torch.multinomial(noise, length, replacement=True)


# --------------------------------------------------AUGMENTATION----------------------------------------

def pfam_hmm_augment(noise, orig_prot_id, prot_id, hmm_dict, fasta_dict, mask):
    seq_probabilities, seq_hit_range = get_seq_hmm_probabilities(qresult=noise[orig_prot_id],
                                                                 hmm_dict=hmm_dict,
                                                                 protein_fasta_dict=fasta_dict,
                                                                 PROT_ID=prot_id)

    # replace 0s with 1s outside seq range to allow multinomial sampling
    seq_probabilities[:seq_hit_range[0]] = 1
    seq_probabilities[seq_hit_range[1]:] = 1
    sampled_aas = torch.multinomial(torch.FloatTensor(seq_probabilities), 1, replacement=True).squeeze()
    # map the AA names to the uniprot21 alphabet
    sampled_aas = Uniprot21().encode(
        str.encode(''.join([PFAM_INDEX_TO_AA[n.item()] for n in sampled_aas])))

    # mask values that are outside the seq range
    mask[:seq_hit_range[0]] = 0
    mask[seq_hit_range[1]:] = 0

    return sampled_aas, mask


def seq_augment(noise, seq, augment_prob, noise_type, orig_prot_id, prot_id, hmm_dict, fasta_dict,
                alphabet):
    # create the random mask... i.e. which positions to infer
    mask = torch.rand(len(seq), device=seq.device)
    mask = (mask < augment_prob).long()

    if noise_type == 'pfam-hmm':
        if noise[orig_prot_id] is None:
            return seq
        sampled_aas, mask = pfam_hmm_augment(noise, orig_prot_id, prot_id, hmm_dict, fasta_dict, mask)

    elif noise_type == 'uniref90':
        sampled_aas = uniref90_agument(noise, len(seq))

    # replace the selected positions with the sampled noise (21 AA)
    # but translate the noise into the dataset's alphabet (21 or less AA groupings)
    # decoded with uniprot because the noise was calculated with uniprot alphabet
    sampled_aas = Uniprot21().decode(sampled_aas)
    sampled_aas = alphabet.encode(sampled_aas)
    seq = (1 - mask) * seq + mask * sampled_aas  # unmasked AA + new values in masked positions

    return seq


# -------------------------------------------FLIP DATASET--------------------------------------------

class FlipDataset(Dataset):
    def __init__(self, base_pth='../../data/FLIP', dataset='aav', split='des_mut', type='train',
                 alphabet=Uniprot21(), augment_prob=0, noise_type='pfam-hmm', full_train=False):

        df = pd.read_csv(f'{base_pth}/{dataset}/splits/{split}.csv')
        # remove '*' which represents deletions in the wild type
        df.sequence = df['sequence'].apply(lambda s: re.sub(r'[^A-Z]', '', s.upper()))
        if type == 'train':
            if full_train:
                df = df[df.set == 'train']
            else:
                df = df[(df.set == 'train') & (df.validation != True)]
        elif type == 'val':
            df = df[(df.set == 'train') & (df.validation == True)]
        elif type == 'test':
            df = df[df.set == 'test']

        df.reset_index(drop=True, inplace=True)

        self.dataset = df
        self.alphabet = alphabet
        self.noise_type = noise_type
        self.augment_prob = augment_prob
        self.seq_id = dataset + '_' + split

        if noise_type == 'pfam-hmm':
            self.hmm_dict = setup_hmm_dict(os.path.dirname(base_pth) + '/')
            with open(f"{base_pth}/pfam_hmm_hits_output.pkl", "rb") as f:
                self.noise = pickle.load(f)
        elif self.noise_type == 'uniref90':
            self.noise = LMDBDataset('/scratch/ii1g17/protein-embeddings/uniref-2018july/uniref90/'
                                     'uniref90.fasta.lmdb').noise

        # fake fasta dict
        self.fasta_dict = {self.seq_id: SeqRecord(Seq(seq_aa), id="") for seq_aa in self.dataset.sequence}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        seq = self.dataset.sequence[idx]
        seq = torch.from_numpy(self.alphabet.encode(str.encode(seq)))
        target = self.dataset.target[idx]

        # TODO: the augment does not work because sequences have different lengths and the nois points to the
        # same hmm prob distributions. We can hit each sequence in turn but I don't think that would do anything
        # as the matched region is going to be the same for all of them.
        if self.augment_prob > 0:
            seq = seq_augment(self.noise, seq, self.augment_prob, self.noise_type, self.seq_id, self.seq_id,
                              self.hmm_dict,
                              self.fasta_dict, self.alphabet)

        return seq, target


# --------------------------------------------------ENZYME-------------------------------------------------------------
class EnzymeDataset(Dataset):

    def __init__(self, enzyme_path, split="train", alphabet=Uniprot21(), augment_prob=0, noise_type='pfam-hmm',
                 filter_dataset=False):
        self.alphabet = alphabet
        self.metadata_path = enzyme_path
        self.augment_prob = augment_prob
        self.uniprot = Uniprot21()
        self.noise_type = noise_type
        self.pdb_fasta = SeqIO.to_dict(SeqIO.parse(f'{enzyme_path}/metadata/pdb_seqres.txt', "fasta"))
        self.name = split

        with open(f'{enzyme_path}/metadata/base_split.json') as json_file:
            json_splits = json.load(json_file)
            if split == "all":
                self.prot_ids = json_splits["train"] + json_splits["valid"] + json_splits["test"]
            else:
                self.prot_ids = json_splits[split]

        if filter_dataset:
            # remove chains that are in both train and test, or  test and valid
            self.prot_ids = [p for p in self.prot_ids if p not in MIXED_SPLIT_CHAINS]

        with open(f"{enzyme_path}/metadata/function_labels.json", "r") as f:
            self.labels_all = json.load(f)

        with open(f"{enzyme_path}/metadata/labels_to_idx.json", "r") as f:
            self.labels_to_idx = json.load(f)

        # used just with the hmm noise type
        self.hmm_dict = None
        if self.noise_type == 'pfam-hmm':
            self.hmm_dict = setup_hmm_dict(os.path.dirname(enzyme_path) + '/')
            with open(f"{enzyme_path}/metadata/pfam_hmm_hits_output.pkl", "rb") as f:
                self.noise = pickle.load(f)
        elif self.noise_type == 'uniref90':
            self.noise = LMDBDataset('/scratch/ii1g17/protein-embeddings/uniref-2018july/uniref90/'
                                     'uniref90.fasta.lmdb').noise

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, index):
        orig_prot_id = self.prot_ids[index]
        prot_id = orig_prot_id if self.prot_ids[index] not in PDB_OBSOLETE_REMAP else PDB_OBSOLETE_REMAP[orig_prot_id]

        prot_record = self.pdb_fasta[prot_id]
        seq = torch.from_numpy(self.alphabet.encode(bytes(prot_record.seq)))

        if self.noise_type is not None:
            seq = seq_augment(self.noise, seq, self.augment_prob, self.noise_type, orig_prot_id, prot_id, self.hmm_dict,
                              self.pdb_fasta, self.alphabet)
        return seq, self.labels_to_idx[self.labels_all[orig_prot_id]], orig_prot_id, prot_record.seq


# ------------------------------------------------FOLD HOMOLOGY DATASET------------------------------------------------
class FoldHomologyDataset(Dataset):
    def __init__(self, dataset_pth='../../data/HomologyTAPE', data_split='train', alphabet=Uniprot21(),
                 augment_prob=0, noise_type='pfam-hmm'):
        self.labels_to_idx = pd.read_csv(f'{dataset_pth}/class_map.txt', delimiter='\t', header=None,
                                         names=['name', 'id']).set_index('name')['id'].to_dict()
        df = pd.read_csv(f'{dataset_pth}/{data_split}.txt', delimiter='\t', header=None, names=['scopid', 'name'])
        self.prot_ids = df['scopid'].values
        self.labels = df['name'].values
        self.alphabet = alphabet
        self.scop_fasta = SeqIO.to_dict(SeqIO.parse(f'{dataset_pth}/astral-scopdom-seqres-gd-sel-gs-bib-95-1.75.fa',
                                                    "fasta"))
        self.name = data_split.split('_')[1].upper() if '_' in data_split else data_split.upper()
        self.augment_prob = augment_prob
        self.noise_type = noise_type

        # used just with the hmm noise type
        self.hmm_dict = None
        if self.noise_type == 'pfam-hmm':
            self.hmm_dict = setup_hmm_dict(f'{os.path.dirname(dataset_pth)}/')
            with open(f"{dataset_pth}/pfam_hmm_hits_output.pkl", "rb") as f:
                self.noise = pickle.load(f)
        elif self.noise_type == 'uniref90':
            self.noise = LMDBDataset('/scratch/ii1g17/protein-embeddings/uniref-2018july/uniref90/'
                                     'uniref90.fasta.lmdb').noise

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, index):
        id = self.prot_ids[index] if self.prot_ids[index] in self.scop_fasta else 'g' + self.prot_ids[index][1:]
        prot_record = self.scop_fasta[id]
        seq = torch.from_numpy(self.alphabet.encode(bytes(prot_record.seq.upper())))
        if self.noise_type is not None:
            seq = seq_augment(self.noise, seq, self.augment_prob, self.noise_type, self.prot_ids[index], id,
                              self.hmm_dict,
                              self.scop_fasta, self.alphabet)
        return seq, self.labels_to_idx[self.labels[index]], self.prot_ids[index], prot_record.seq


# -----------------------------------------------DATASETS FOR PROTEIN SEQUENCES---------------------------------------
class LMDBDataset:
    """
    Adapted from: https://github.com/songlab-cal/tape
    Creates a dataset from an lmdb file.
    Args:
        data_file: Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self, data_file, max_length=500, alphabet=Uniprot21(), in_memory=False, delim="|DELIM|",
                 output=sys.stdout):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get("nr_seq".encode()).decode())
            self.noise = torch.from_numpy(pickle.loads(txn.get("marginal_distribution".encode())))
            self.lengths = pickle.loads(txn.get("seq_lengths".encode()))

            print("Number of sequences in the Unirep90 .lmdb dataset: ", self._num_examples,
                  file=output)

        if in_memory:
            cache = [None] * self._num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self.max_length = max_length
        self.alphabet = alphabet
        self.delim = delim

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = txn.get(str(index).encode()).decode()
                if self._in_memory:
                    self._cache[index] = item

        description, seq = item.split(self.delim)
        seq = torch.from_numpy(self.alphabet.encode(seq.encode()))
        if 0 < self.max_length < len(seq):
            # randomly sample a subsequence of length max_length
            j = random.randint(0, len(seq) - self.max_length)
            seq = seq[j:j + self.max_length]

        return seq.long()


class ClozeDataset:
    """
    Wrapper for LMDBDataset
    Args:
        dataset (LMDBDataset): dataset to wrap
        probability (float): number in [0,1] that represents the chance of masking e.g. 0.1 means 10%
        noise (torch.FloatTensor): the background marginal distribution of the dataset, len(noise) == nr_AA
    """

    def __init__(self, dataset, probability, noise):
        self.dataset = dataset
        self.probability = probability
        self.noise = noise

        self.uniprot = Uniprot21()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """"
        Returns:
            list: protein's sequence as a list of integers highlighting the AA number
            list: protein's labels as a list of integers: original values for masked AAs and unknown 20 for the rest """

        item = self.dataset[i]
        n = len(self.noise)  # number of tokens

        # create the random mask... i.e. which positions to infer
        mask = torch.rand(len(item), device=item.device)
        mask = (mask < self.probability).long()  # we mask with probability p

        # keep original AA label for masked positions and assign unknown AA label to all the others
        labels = mask * item + (1 - mask) * (n - 1)

        # sample the masked positions from the noise distribution
        noise = torch.multinomial(self.noise, len(item), replacement=True)

        # replace the masked positions with the sampled noise (21 AA)
        # but translate the noise into the dataset's alphabet (21 or less AA groupings)
        # decoded with uniprot because the noise was calculated with uniprot alphabet
        noise = self.uniprot.decode(noise)
        noise = self.dataset.alphabet.encode(noise)

        x = (1 - mask) * item + mask * noise  # unmasked AA + new values in masked positions

        # eg: x = [4,5,13,19], labels = [4,20,20,13]
        # (only the first and last AA have been replaced with noise AA)
        return x, labels
