import argparse
import json
import os
import re
import sys

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from Bio import SeqIO
from tqdm import tqdm

from proemb.utils.hmm_util import run_hmmscan
import pickle as pkl
import pandas as pd

PDB_OBSOLETE_REMAP = {'6fed_C': '6eyd_C',
                      '5jn1_A': '6uzu_A',
                      '5xwq_A': '7fbt_A',
                      '4ror_A': '5ulv_A',
                      '6ihz_A': '7emn_A',
                      '6ihz_B': '7emn_B',
                      '6fed_E': '6eyd_E',
                      '6gt1_C': '6s73_C',
                      '6gt1_D': '6s73_D',
                      '6gt1_B': '6s73_B',
                      '6gt1_C': '6s73_C',
                      '6gt1_A': '6s73_A',
                      '6fed_B': '6eyd_B',
                      '6fed_A': '6eyd_A',
                      '4wto_A': '6kjb_A',
                      '1bvs_D': '7oa5_D',
                      '1bvs_A': '7oa5_A',
                      '1bvs_C': '7oa5_C',
                      '1bvs_E': '7oa5_E',
                      '1bvs_H': '7oa5_H',
                      '5x8o_A': '6lk4_A',
                      '5urr_D': '7mtx_D',
                      '3ohm_B': '7sq2_B',
                      '5hbg_B': '6ahi_B',
                      '5hbg_A': '6ahi_A',
                      '5obl_A': '7obe_A',
                      '5obl_B': '7obe_B',
                      '6fed_D': '6eyd_D',

                      '2i6l_A': '7aqb_A',
                      '2i6l_B': '7aqb_B',
                      '3r5q_A': '7ve3_A'
                      }

MIXED_SPLIT_CHAINS = ['4y84_X', '5l5e_X', '6huu_J', '4qby_J', '4ya9_J', '5mp9_k', '5mpa_k', '3von_E', '3von_b',
                      '3von_p', '3von_i', '6hed_4',
                      '6hec_5', '6he8_4', '6he9_3', '6he7_6', '6he8_k', '6hed_h', '6hea_i', '6hea_h', '6he9_i',
                      '3mg8_I', '4qlq_W', '6huv_I',
                      '5fga_W', '4qby_W', '5mpa_j', '5mp9_j', '5lf1_b', '5lf1_B', '5gjq_j', '1iru_R', '5gjq_k',
                      '5lf0_W', '5m32_I', '5le5_I',
                      '5lf1_I', '5lf3_I', '5gjq_q']



def write_enzyme_fasta(root='../../data/enzyme'):
    # Opening JSON file
    with open(f'{root}/metadata/base_split.json') as json_file:
        data = json.load(json_file)

    # all the proteins we need to generate fasta files for
    data = data['train'] + data['valid'] + data['test']

    # all known pdb chains in fasta format
    all_pdb_fasta = SeqIO.to_dict(SeqIO.parse(f'{root}/metadata/pdb_seqres.txt', "fasta"), lambda rec: rec.id)

    for id in tqdm(data, total=len(data)):
        prot_id = id if id not in PDB_OBSOLETE_REMAP else PDB_OBSOLETE_REMAP[id]
        record = all_pdb_fasta[prot_id]

        filename = f'{root}/fasta/{prot_id}.fasta'
        with open(filename, "w") as handle:
            SeqIO.write(record, handle, "fasta")

def write_enzyme_hmm(root='../../../data/'):

    with open(f'{root}enzyme/metadata/base_split.json') as json_file:
        data = json.load(json_file)

    # all the proteins we need to generate fasta files for
    data = data['train'] + data['valid'] + data['test']

    pdb_fasta_dict = SeqIO.to_dict(SeqIO.parse(f'{root}enzyme/metadata/pdb_seqres.txt', "fasta"))
    hmm_output = {}
    for id in tqdm(data, total=len(data)):
        prot_id = id if id not in PDB_OBSOLETE_REMAP else PDB_OBSOLETE_REMAP[id]
        qresult = run_hmmscan(prot_id, pdb_fasta_dict, root)

        if not qresult.hits:
            print(f"Skipping {prot_id} because no hits were found")
            hmm_output[id] = None
            continue
        hmm_output[id] = qresult

    # save the hmm output
    with open(f'{root}enzyme/metadata/pfam_hmm_hits_output.pkl', 'wb') as f:
        pkl.dump(hmm_output, f)


def write_homology_hmm(root='../../../data'):

    df = pd.read_csv(f'{root}/HomologyTAPE/training.txt', delimiter='\t', header=None, names=['scopid', 'name'])
    data = df['scopid'].values
    scop_fasta = SeqIO.to_dict(SeqIO.parse(f'{root}/HomologyTAPE/astral-scopdom-seqres-gd-sel-gs-bib-95-1.75.fa',
                                           "fasta"))
    hmm_output = {}
    for id in tqdm(data, total=len(data)):
        prot_id = id if id in scop_fasta else 'g' + id[1:]
        qresult = run_hmmscan(prot_id, scop_fasta, root)
        if not qresult.hits:
            print(f"Skipping {prot_id} because no hits were found")
            hmm_output[id] = None
            continue
        hmm_output[id] = qresult

    # save the hmm output
    with open(f'{root}HomologyTAPE/pfam_hmm_hits_output.pkl', 'wb') as f:
        pkl.dump(hmm_output, f)

def write_flip_hmm(base_pth='../../../data/FLIP'):

    hmm_output = {}
    for (dataset, split) in [('aav', 'des_mut'), ('aav', 'low_vs_high'), ('aav', 'mut_des'), ('aav','one_vs_many'),
                             ('aav', 'seven_vs_many'), ('aav', 'two_vs_many'),
                             ('gb1', 'low_vs_high'), ('gb1', 'one_vs_rest'), ('gb1', 'three_vs_rest'),
                             ('gb1', 'two_vs_rest'),
                             ('meltome', 'human'), ('meltome', 'human_cell')]:

        df = pd.read_csv(f'{base_pth}/{dataset}/splits/{split}.csv')
        # remove '*' which represents deletions in the wild type
        df.sequence = df['sequence'].apply(lambda s: re.sub(r'[^A-Z]', '', s.upper()))

        print("Dateset: ", dataset, " Split: ", split)

        # all the sequences are the same apart from the mutations
        seq = df.sequence[0]
        id = dataset + '_' + split
        seq_record = SeqRecord(Seq(seq), id=id)
        qresult = run_hmmscan("No-id", None, '../../../data/', seq_record)

        if not qresult.hits:
            print(f"Skipping {seq} because no hits were found")
            hmm_output[id] = None
            continue
        hmm_output[id] = qresult
        print(qresult)

    with open(f'{base_pth}/pfam_hmm_hits_output.pkl', 'wb') as f:
        pkl.dump(hmm_output, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script for generating fasta files')
    parser.add_argument('--root', default='../../../data/')
    args = parser.parse_args()
    #write_enzyme_fasta(root=args.root)
    #write_enzyme_hmm(root=args.root)

    #write_homology_hmm(root=args.root)

    write_flip_hmm()
