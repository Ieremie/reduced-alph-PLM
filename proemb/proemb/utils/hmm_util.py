import os

import numpy as np
from Bio import SearchIO, SeqIO
from pyhmmer.plan7 import HMMFile
import subprocess

def run_hmmscan(base, protein_seq, PROT_ID):
    """
    Runs hmmscan on a protein sequence and returns the qresult object (parsed hmmscan output).

    Args:   PROT_ID (str): protein ID
            protein_fasta_dict (dict): dictionary of protein sequences, keys are protein IDs
            base (str): path to the proemb directory

    Returns:
        qresult (biopython parsing object): hmmscan output
    """

    # save SCOPid seq as fasta
    SeqIO.write(protein_seq, f"{PROT_ID}.fasta", "fasta")

    # RUN HMMSCAN using subprocess
    hmm_scan = subprocess.Popen(
        ["hmmscan", "-o", f"{PROT_ID}.hmmscan", f"{base}pfam-hmm/Pfam-A.hmm", f"{PROT_ID}.fasta"])
    # wait until hmmscan is done
    hmm_scan.wait()

    qresult = SearchIO.read(f'{PROT_ID}.hmmscan', 'hmmer3-text')

    # remove fasta and hmmscan files
    os.remove(f"{PROT_ID}.fasta")
    os.remove(f"{PROT_ID}.hmmscan")

    return qresult


def setup_hmm_dict(base):
    """
    Returns a dictionary of HMMs, keys are HMM names.

    Args:   base (str): path to the proemb directory

    Returns:
        hmm_dict (dict): dictionary of HMMs, keys are HMM names
    """

    with HMMFile(f"{base}pfam-hmm/Pfam-A.hmm") as hmm_file:
        hmms = list(hmm_file)
    hmm_dict = {hmm.name: hmm for hmm in hmms}

    return hmm_dict

def get_seq_hmm_probabilities(qresult, hmm_dict, protein_fasta_dict, PROT_ID):
    """
    Returns the probability distribution of the 20 amino acids for each residue in the protein sequence.
    The probability distributions are taken from the HMM match/inseration states.

    Args:   qresult (biopython parsing object): hmmscan output
            hmm_dict (dict): dictionary of HMMs, keys are HMM names
            protein_fasta_dict (dict): dictionary of protein sequences, keys are protein IDs
            PROT_ID (str): protein ID

    Returns:
        seq_probabilities (np.array): probability distribution of the 20 amino acids for each residue in the protein sequence

    """

    # choosing the first fragment from the domain hit with the lowest bit score
    hps_fragment = qresult.fragments[0]
    hmm_name = hps_fragment.hit_id

    hmm_hit_range = hps_fragment.hit_range
    hmm_hit = hps_fragment.hit.seq

    seq_hit_range = hps_fragment.query_range
    seq_hit = hps_fragment.query.seq

    hmm = hmm_dict[hmm_name.encode()]
    seq_probabilities = np.zeros((len(protein_fasta_dict[PROT_ID].seq), 20))

    seq_prob_index = seq_hit_range[0]
    hmm_hit_index = hmm_hit_range[0]

    for hit_index in range(len(seq_hit)):

        # ignore gaps (sequence deletions compared to the hmm)
        if seq_hit[hit_index] == '-':
            # we go to the next state in the hmm
            hmm_hit_index += 1
            continue

        # match
        if hmm_hit[hit_index] != '.':
            # +1 because the first column is the start state
            seq_probabilities[seq_prob_index] = hmm.match_emissions[hmm_hit_index + 1]
            hmm_hit_index += 1
        else:
            # insert, we don't go to the next state in the hmm
            seq_probabilities[seq_prob_index] = hmm.insert_emissions[hmm_hit_index + 1]

        seq_prob_index += 1

    return seq_probabilities, seq_hit_range