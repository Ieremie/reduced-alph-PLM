from __future__ import print_function, division

import sys

from Bio import SeqIO
from pathlib import Path
import os
import lmdb
import tqdm
import numpy as np
from proemb.alphabets import Uniprot21
import pickle

import argparse
import os

def parse(f, comment=b'#', pbar=None):
    names = []
    sequences = []
    name = None
    sequence = []

    for line in f:

        # update the progress bar
        if pbar is not None:
            pbar.update(len(line))

        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(b'>'):
            if name is not None:
                names.append(name)
                sequences.append(b''.join(sequence))
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if name is not None:
        names.append(name)
        sequences.append(b''.join(sequence))

    return names, sequences


def fasta_to_lmdb(path, save_path):
    """
    Creates a .lmdb file from an a fasta file.
    It calculates the AA's marginal distribution and protein lengths
    Args:
        path: Path to fasta file.
        save_path: save the dataset to a different path
    """

    data_file = Path(path)
    if not data_file.exists():
        raise FileNotFoundError(data_file)

    size = os.path.getsize(data_file)
    # increasing the size to make sure lmdb does not complain
    # 63GB in .fasta <=> 104GB in .lmdb format
    map_size = int(size * 2)
    print("Creating lmdb file with size: ", map_size / pow(1024, 3), "GB")
    print("NOTE: progress bar is not accurate due to SeqIO parsing")

    counts = np.zeros(21)
    lengths = []
    alph = Uniprot21()
    env = lmdb.open(save_path + '.lmdb', map_size=map_size)
    with env.begin(write=True) as txn:
        nr_seq = 0
        with tqdm.tqdm(total=size) as pbar:
            for record in SeqIO.parse(str(data_file), 'fasta'):
                pbar.update(len(record))
                protein = record.description + "|DELIM|" + str(record.seq)
                txn.put(str(nr_seq).encode(), protein.encode())
                nr_seq += 1

                # calculating marginal distribution
                encoded_seq = alph.encode(bytes(record.seq))
                v, c = np.unique(encoded_seq, return_counts=True)
                counts[v] = counts[v] + c

                # saving lengths for future use
                lengths.append(len(record.seq))

        noise = counts / counts.sum()
        print('# amino acid marginal distribution:', noise, file=sys.stderr)
        txn.put("nr_seq".encode(), str(nr_seq).encode())
        txn.put("marginal_distribution".encode(), pickle.dumps(noise))
        txn.put("seq_lengths".encode(), pickle.dumps(lengths))

def main():

    print(os.getcwd())
    parser = argparse.ArgumentParser('Script for transforming fasta file to lmdb')
    parser.add_argument('--path-fasta', default='../data/uniprot/uniref90.fasta',
                        help='path to training dataset in fasta format (default: data/uniprot/uniref90.fasta)')
    parser.add_argument('--save-path', default='/scratch/ii1g17/protein-embeddings/uniref90.fasta',
                        help='path to save the lmdb file, NOT BACKED UP, TEMP FILE!'
                             ' (default: /scratch/ii1g17/protein-embeddings/uniref90.fasta)')
    parser.add_argument('--convert-fasta-to-lmdb', action='store_true', help='convert fasta to lmdb file')
    args = parser.parse_args()

    if args.convert_fasta_to_lmdb:
        fasta_to_lmdb(args.path_fasta, args.save_path)


if __name__ == '__main__':
    main()
