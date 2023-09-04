"""
Adapted from https://github.com/tbepler/prose
"""

from __future__ import print_function, division
import random
import numpy as np

PFAM_AA_TO_INDEX = dict(A=0, C=1, D=2, E=3, F=4, G=5, H=6, I=7, K=8, L=9, M=10, N=11, P=12, Q=13, R=14, S=15, T=16,
                        V=17, W=18, Y=19)
PFAM_INDEX_TO_AA = {v: k for k, v in PFAM_AA_TO_INDEX.items()}


class Alphabet:
    def __init__(self, encoding, missing=20):
        # mapping the char -> ASCII value
        self.chars = np.frombuffer(b'ARNDCQEGHILKMFPSTWYVXOUBZ', dtype=np.uint8)

        # we assume that any wrong AA label is mapped to missing type 20
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        self.encoding[self.chars] = encoding
        self.size = encoding.max() + 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x):
        """ encode a byte string into alphabet indices """
        x = np.frombuffer(x, dtype=np.uint8)
        return self.encoding[x]

    def decode(self, x):
        """ decode index array, x, to byte string of this alphabet """
        string = self.chars[x]
        return string.tobytes()

    def decode_reduced(self, x, select="first"):

        inverted_groupings = {}
        for key, value in self.groupings.items():
            if key not in ["O", "U", "B", "Z"]:
                inverted_groupings[value] = inverted_groupings.get(value, []) + [key]

        # if random then we encode a random amino acid from the cluster
        if select == "random":
            decoded = [np.random.choice(inverted_groupings[i]) for i in x]

        elif select == "first":
            decoded = [inverted_groupings[i][0] for i in x]

        # we encode a random amino acid from the cluster, but we keep the same random amino acid along the sequence
        elif select == "random_fixed":
            fixed_idx = {k: random.randint(0, len(v) - 1) for k, v in inverted_groupings.items()}
            decoded = [inverted_groupings[i][fixed_idx[i]] for i in x]

        # ecode str to bytes
        return "".join(decoded).encode()


class Uniprot21(Alphabet):
    def __init__(self):
        chars = 'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))

        # O as K
        # U as C
        # B and Z as X (unknown)
        encoding[21:] = [11, 4, 20, 20]  # O, U, B, Z
        self.groupings = {k: v for k, v in zip(chars, encoding)}
        super(Uniprot21, self).__init__(encoding=encoding, missing=20)


class Uniprot21_Reduced_1(Alphabet):
    """
    Reducing uniprot based on the clusters found during training
    """

    def __init__(self):
        chars = 'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))

        # O as K
        # U as C
        # B and Z as X (unknown)
        encoding[21:] = [11, 4, 20, 20]  # O, U, B, Z
        # PRO = GLU
        encoding[14] = 6

        self.groupings = {k: v for k, v in zip(chars, encoding)}
        super(Uniprot21_Reduced_1, self).__init__(encoding=encoding, missing=20)


class Uniprot21_Reduced_2(Alphabet):
    """
    Reducing uniprot based on the clusters found during training
    """

    def __init__(self):
        chars = 'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))

        # O as K
        # U as C
        # B and Z as X (unknown)
        encoding[21:] = [11, 4, 20, 20]  # O, U, B, Z
        # PRO = GLU
        encoding[14] = 6

        # LEU = HIS
        encoding[10] = 8

        self.groupings = {k: v for k, v in zip(chars, encoding)}
        super(Uniprot21_Reduced_2, self).__init__(encoding=encoding, missing=20)


class SDM12(Alphabet):
    """
    A D KER N TSQ YF LIVM C W H G P
    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2732308/#B33
    "Reduced amino acid alphabets exhibit an improved sensitivity and selectivity in fold assignment"
    Peterson et al. 2009. Bioinformatics.

    The SDM12 alphabet maintains clusters for acidic/basic (KER), polar
    (TSQ), aromatic (YF) and mostly aliphatic (LIVM) groups.Two
    non-intuitive results in these groupings are the omission of aspartic
    acid from the acidic/basic
    """

    def __init__(self):
        chars = 'ARNDCQEGHILKMFPSTWYVXOUBZ'

        # keep XBZ as unknown with label 20
        self.groupings = {'A': 0,
                          'D': 1,
                          'K': 2, 'E': 2, 'R': 2,
                          'N': 3,
                          'T': 4, 'S': 4, 'Q': 4,
                          'Y': 5, 'F': 5,
                          'L': 6, 'I': 6, 'V': 6, 'M': 6,
                          'C': 7,
                          'W': 8,
                          'H': 9,
                          'G': 10,
                          'P': 11,

                          # unknown or ambiguous in uniprot
                          'O': 2,
                          'U': 7,
                          'X': 20, 'B': 20, 'Z': 20}

        encoding = np.array([self.groupings[aa_letter] for aa_letter in chars])
        super(SDM12, self).__init__(encoding=encoding, missing=20)


class GBMR4(Alphabet):
    """
    ADKERNTSQ YFLIVMCWH G P

    the GBMR4, alphabet glycine and proline are singled
    out as being structurally dissimilar from the other amino acids;
    the remaining two groups reflect a hydrophobic (YFLIVMCWH)
    and polar (ADKERNTSQ) classification
    """

    def __init__(self):
        chars = 'ARNDCQEGHILKMFPSTWYVXOUBZ'

        self.groupings = {'A': 0, 'D': 0, 'K': 0, 'E': 0, 'R': 0, 'N': 0, 'T': 0, 'S': 0, 'Q': 0,
                          'Y': 1, 'F': 1, 'L': 1, 'I': 1, 'V': 1, 'M': 1, 'C': 1, 'W': 1, 'H': 1,
                          'G': 2,
                          'P': 3,

                          # unknown or ambiguous in uniprot
                          'O': 0,
                          'U': 1,
                          'X': 20, 'B': 20, 'Z': 20}

        encoding = np.array([self.groupings[aa_letter] for aa_letter in chars])
        super(GBMR4, self).__init__(encoding=encoding, missing=20)


class GBMR7(Alphabet):
    """
    GLY (G)
    ASP (D) = ASN (N)
    ALA (A) = GLU (E) = PHE (F) = ILE (I) = LYS (K) = LEU (L) = MET (M) = GLN (Q) = ARG (R) = VAL (V) = TRP (W) = TYR (Y)
    CYS (C) = HIS (H)
    THR (T)
    SER (S)
    PRO (P)
    """

    def __init__(self):
        chars = 'ARNDCQEGHILKMFPSTWYVXOUBZ'

        self.groupings = {'G': 0,
                          'D': 1, 'N': 1,
                          'A': 2, 'E': 2, 'F': 2, 'I': 2, 'K': 2, 'L': 2, 'M': 2, 'Q': 2, 'R': 2, 'V': 2, 'W': 2,
                          'Y': 2,
                          'C': 3, 'H': 3,
                          'T': 4,
                          'S': 5,
                          'P': 6,

                          # unknown or ambiguous in uniprot
                          'O': 2,
                          'U': 3,
                          'X': 20, 'B': 20, 'Z': 20}

        encoding = np.array([self.groupings[aa_letter] for aa_letter in chars])

        super(GBMR7, self).__init__(encoding=encoding, missing=20)


class HSDM17(Alphabet):
    """A D KE R N T S Q Y F LIV M C W H G P
    In HSDM17,
    only the strongest associations among these are maintained: acidic/basic (KE) and aliphatic (LIV)."""

    def __init__(self):
        chars = 'ARNDCQEGHILKMFPSTWYVXOUBZ'

        self.groupings = {'A': 0,
                          'D': 1,
                          'K': 2, 'E': 2,
                          'R': 3,
                          'N': 4,
                          'T': 5,
                          'S': 6,
                          'Q': 7,
                          'Y': 8,
                          'F': 9,
                          'L': 10, 'I': 10, 'V': 10,
                          'M': 11,
                          'C': 12,
                          'W': 13,
                          'H': 14,
                          'G': 15,
                          'P': 16,

                          # unknown or ambiguous in uniprot
                          'O': 2,
                          'U': 12,
                          'X': 20, 'B': 20, 'Z': 20}

        encoding = np.array([self.groupings[aa_letter] for aa_letter in chars])
        super(HSDM17, self).__init__(encoding=encoding, missing=20)


class WASS_275_LEVEL_1_30BINS(Alphabet):
    """
    TRP (W) = MET (M)
    ASP (D) = ILE (I)
    PRO (P)
    CYS (C)
    ALA (A) = VAL (V)
    LYS (K)
    THR (T)
    ARG (R) = GLU (E)
    GLY (G)
    LEU (L)
    TYR (Y)
    SER (S) = HIS (H)
    PHE (F)
    ASN (N) = GLN (Q)
    """

    def __init__(self):
        chars = 'ARNDCQEGHILKMFPSTWYVXOUBZ'

        self.groupings = {'W': 0, 'M': 0,
                          'D': 1, 'I': 1,
                          'P': 2,
                          'C': 3,
                          'A': 4, 'V': 4,
                          'K': 5,
                          'T': 6,
                          'R': 7, 'E': 7,
                          'G': 8,
                          'L': 9,
                          'Y': 10,
                          'S': 11, 'H': 11,
                          'F': 12,
                          'N': 13, 'Q': 13,

                          # unknown or ambiguous in uniprot
                          # O as K
                          # U as C
                          # B and Z as X (unknown)
                          'O': 5,
                          'U': 3,
                          'X': 20, 'B': 20, 'Z': 20}

        encoding = np.array([self.groupings[aa_letter] for aa_letter in chars])
        super(WASS_275_LEVEL_1_30BINS, self).__init__(encoding=encoding, missing=20)


class MMSEQS2(Alphabet):
    """
    The default alphabet
    with A = 13, which performed well over all tested clustering sequence identities
    from 50% to 100%, merges (L, M), (I, V), (K, R), (E, Q), (A, S, T), (N, D),
    and (F, Y)

    L = M
    I = V
    K = R
    E = Q
    A = S = T
    N = D
    F = Y

    C
    G
    H
    P
    W

    """

    def __init__(self):
        chars = 'ARNDCQEGHILKMFPSTWYVXOUBZ'

        self.groupings = {'L': 0, 'M': 0,
                          'I': 1, 'V': 1,
                          'K': 2, 'R': 2,
                          'E': 3, 'Q': 3,
                          'A': 4, 'S': 4, 'T': 4,
                          'N': 5, 'D': 5,
                          'F': 6, 'Y': 6,

                          'C': 7,
                          'G': 8,
                          'H': 9,
                          'P': 10,
                          'W': 11,

                          # unknown or ambiguous in uniprot
                          # O as K
                          # U as C
                          # B and Z as X (unknown)
                          'O': 2,
                          'U': 7,
                          'X': 20, 'B': 20, 'Z': 20}

        encoding = np.array([self.groupings[aa_letter] for aa_letter in chars])
        super(MMSEQS2, self).__init__(encoding=encoding, missing=20)


class WWMJ(Alphabet):
    """
    A computational approach to simplifying the protein folding alphabet(1999)

    CMFILVWY
    ATH
    GP
    DE
    SNQRK
    """

    def __init__(self):
        chars = 'ARNDCQEGHILKMFPSTWYVXOUBZ'

        self.groupings = {'C': 0, 'M': 0, 'F': 0, 'I': 0, 'L': 0, 'V': 0, 'W': 0, 'Y': 0,
                          'A': 1, 'T': 1, 'H': 1,
                          'G': 2, 'P': 2,
                          'D': 3, 'E': 3,
                          'S': 4, 'N': 4, 'Q': 4, 'R': 4, 'K': 4,

                          # unknown or ambiguous in uniprot
                          # O as K
                          # U as C
                          # B and Z as X (unknown)
                          'O': 4,
                          'U': 0,
                          'X': 20, 'B': 20, 'Z': 20}
        encoding = np.array([self.groupings[aa_letter] for aa_letter in chars])
        super(WWMJ, self).__init__(encoding=encoding, missing=20)


# get the alphabet based on string name
def get_alphabet(name):
    if name == 'uniprot21':
        return Uniprot21()
    elif name == 'uniprot21_reduced_1':
        return Uniprot21_Reduced_1()
    elif name == 'uniprot21_reduced_2':
        return Uniprot21_Reduced_2()
    elif name == 'sdm12':
        return SDM12()
    elif name == 'gbmr4':
        return GBMR4()
    elif name == 'gbmr7':
        return GBMR7()
    elif name == 'hsdm17':
        return HSDM17()
    elif name == 'wass_275_level_1_30bins':
        return WASS_275_LEVEL_1_30BINS()
    elif name == 'mmseqs2':
        return MMSEQS2()
    elif name == 'wwmj':
        return WWMJ()
    else:
        raise ValueError('Unknown alphabet: {}'.format(name))
