import argparse

import numpy as np
import torch
from Bio import SeqIO
from torch.nn.utils.rnn import pad_sequence
import tqdm

from models.lstm import SkipLSTM
from models.multitask import ProSEMT
from proemb.alphabets import get_alphabet


def load_model(model_name, model_path="../data/saved-models/release-reduced-alphabet"):

    print("Loading model...")
    model_path = f"{model_path}/{model_name}_iter_240000_checkpoint.pt"
    encoder = SkipLSTM(21, 100, 512, 3)
    model = ProSEMT(encoder, None, None, None, None)
    # only using the encoder weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'], strict=False)
    print("Model loaded.")

    return model

def load_fasta(fasta_path):
    print("Loading fasta...")
    fasta_seqs = SeqIO.parse(f"{fasta_path}", "fasta")
    fasta_seqs = SeqIO.to_dict(fasta_seqs)
    seqs = [str(v.seq) for k,v in fasta_seqs.items()]
    print("Fasta loaded.")

    return seqs
def embed_prot_seq(fasta_path, model, batch_size, alphabet_name):

    seqs = load_fasta(fasta_path)
    alphabet = get_alphabet(alphabet_name)

    embeddings = []
    for seqs_batch in tqdm.tqdm(np.array_split(seqs, int(len(seqs) / batch_size))):

        encoded_seqs = [torch.LongTensor(alphabet.encode(str.encode(seq.upper()))) for seq in list(seqs_batch)]
        encoded_seqs = sorted(encoded_seqs, key=lambda x: x.size()[0], reverse=True)
        seqs_lengths = torch.FloatTensor([len(seq) for seq in encoded_seqs])

        encoded_seqs = pad_sequence(encoded_seqs, batch_first=True, padding_value=20)
        embedding = model(encoded_seqs, seqs_lengths, apply_proj=False)
        embeddings.append(embedding.detach().numpy())

    np.save(f'embeddings.npy', embeddings)


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_path', type=str, default='../data/SCOPe/'
                                                          'astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.fa')
    parser.add_argument('--model_name', type=str, default="LM")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--alphabet_name", type=str, default="uniprot21")
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    args = vars(args)

    model = load_model(args['model_name']).to(args['device'])
    embed_prot_seq(args['fasta_path'], model, args['batch_size'], args['alphabet_name'])