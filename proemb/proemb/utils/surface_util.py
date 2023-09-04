import numpy as np
from Bio import pairwise2
from Bio.PDB import Selection, ShrakeRupley, Select
from Bio.PDB.ResidueDepth import residue_depth, ca_depth

# based on the ESM paper plot
AA_PROP = {'A': ['SMALL', 'HYDROPHOBIC'],
           'C': ['MEDIUM', 'UNIQUE'],
           'D': ['MEDIUM', 'NEGATIVELY CHARGED'],
           'E': ['MEDIUM', 'NEGATIVELY CHARGED'],
           'F': ['LARGE', 'AROMATIC'],
           'G': ['SMALL', 'UNIQUE'],
           'H': ['LARGE', 'POSITIVELY CHARGED'],
           'I': ['MEDIUM', 'HYDROPHOBIC'],
           'K': ['MEDIUM', 'POSITIVELY CHARGED'],
           'L': ['MEDIUM', 'HYDROPHOBIC'],
           'M': ['MEDIUM', 'HYDROPHOBIC'],
           'N': ['MEDIUM', 'POLAR'],
           'P': ['SMALL', 'UNIQUE'],
           'Q': ['MEDIUM', 'POLAR'],
           'R': ['LARGE', 'POSITIVELY CHARGED'],
           'S': ['SMALL', 'POLAR'],
           'T': ['SMALL', 'POLAR'],
           'V': ['SMALL', 'HYDROPHOBIC'],
           'W': ['LARGE', 'AROMATIC'],
           'Y': ['LARGE', 'AROMATIC'],

           'UNK': ['UNK', 'UNK']}


ROSE_AA = {'CYS': 'very hydrophobic',

           'LEU': 'hydrophobic',
           'ILE': 'hydrophobic',
           'PHE': 'hydrophobic',
           'TRP': 'hydrophobic',
           'VAL': 'hydrophobic',
           'MET': 'hydrophobic',

           'HIS': 'moderately hydrophobic',
           'TYR': 'moderately hydrophobic',
           'ALA': 'moderately hydrophobic',
           'GLY': 'moderately hydrophobic',
           'THR': 'moderately hydrophobic',

           'SER': 'hydrophilic',
           'PRO': 'hydrophilic',
           'ARG': 'hydrophilic',
           'ASN': 'hydrophilic',
           'GLN': 'hydrophilic',
           'ASP': 'hydrophilic',
           'GLU': 'hydrophilic',

           'LYS': 'very hydrophilic',

           'UNK': 'UNK',

           'ALL': 'ALL',
           'XAA': 'UNK'
           }

# hydrophobicity of AA given by the Rose et al. scale

# SURFACE generated with not HETATM, with disordered atoms and with probe radius 1.4
# SASA normalization based on radii_dict={"D": 1.550}, same as in Biopython RADII dict
SASA_NORM = {'A': 193.21614146941283,
             'C': 207.8091660709734,
             'D': 233.9434470376561,
             'E': 292.5264091428074,
             'F': 296.2057921932092,
             'G': 167.25938562039912,
             'H': 269.7121130269558,
             'I': 262.7082840641545,
             'K': 303.39920998946934,
             'L': 258.36387611447014,
             'M': 276.3388252076202,
             'N': 241.8221969881115,
             'P': 238.3562537425945,
             'Q': 275.9166831195719,
             'R': 328.9515172002599,
             'S': 207.5936402485665,
             'T': 224.5122979269853,
             'V': 242.13572793493975,
             'W': 323.1293920309212,
             'X': 171.74525320408935,
             'Y': 328.43207370454473,

             # max sasa of all residues
             'ALL': 328.43207370454473}

# SURFACE generated with not HETATM, with disordered atoms and with probe radius 1.4
DIST_TO_SURF_NORM = {'A': 13.863261427147128,
                     'C': 13.262251373057197,
                     'D': 14.253642332336645,
                     'E': 14.24767379760616,
                     'F': 12.658830624111337,
                     'G': 14.93631769299833,
                     'H': 12.398553571503074,
                     'I': 12.654537744767747,
                     'K': 12.90096394335276,
                     'L': 12.49909386243177,
                     'M': 12.970166445868625,
                     'N': 14.45986547372234,
                     'P': 12.633685213659392,
                     'Q': 14.278710236448903,
                     'R': 12.306455428090834,
                     'S': 14.235233993105815,
                     'T': 13.941587674138953,
                     'V': 14.274915591144444,
                     'W': 13.692514639006022,
                     'X': 10.09826061495768,
                     'Y': 12.448058907691305,

                     # maximum of all the above
                     'ALL': 14.93631769299833}

DIST_TO_IFACE_NORM = {'A': 138.98491660295704,
                      'C': 138.46398056971643,
                      'D': 139.77299088806717,
                      'E': 149.3321393654664,
                      'F': 143.94800825119106,
                      'G': 150.06224814998217,
                      'H': 152.1444868659457,
                      'I': 144.40705501128605,
                      'K': 145.10715536601506,
                      'L': 147.49960318575722,
                      'M': 127.2585462742218,
                      'N': 143.2491433575261,
                      'P': 141.10341561044248,
                      'Q': 146.29643156770834,
                      'R': 137.65017995806076,
                      'S': 147.1558290560849,
                      'T': 144.9088275155274,
                      'V': 147.27690469355372,
                      'W': 142.52152805002223,
                      'X': 16.126359981777533,
                      'Y': 152.12293093138055,

                      'ALL': 152.1444868659457}

# Kyte Doolittle scale
kd_scale = {"ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9, "ALA": 1.8, "GLY": -0.4,
            "THR": -0.7, "SER": -0.8, "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5,
            "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,

            # unknown AA (umbiguous)
            "ASX": 0, "XAA": 0, "GLX": 0, "XLE": 0, "SEC": 0, "PYL": 0}

protein_letters_1to3 = dict(A="ALA", C="CYS", D="ASP", E="GLU", F="PHE", G="GLY", H="HIS", I="ILE", K="LYS", L="LEU",
                            M="MET", N="ASN", P="PRO", Q="GLN", R="ARG", S="SER", T="THR", V="VAL", W="TRP", Y="TYR")

protein_letters_1to3_extended = {
    **protein_letters_1to3,
    **{"B": "ASX", "X": "XAA", "Z": "GLX", "J": "XLE", "U": "SEC", "O": "PYL"},
}
protein_letters_3to1_extended = {value: key for key, value in protein_letters_1to3_extended.items()}

# we encode B and Z as unknown X to be the same encoding as the alphabet used in the sequence training
# we keep the alphabet to 21 characters
protein_letters_3to1_extended_grouped = {k: v for k, v in protein_letters_3to1_extended.items()}
protein_letters_3to1_extended_grouped['ASX'] = 'X'
protein_letters_3to1_extended_grouped['GLX'] = 'X'
protein_letters_3to1_extended_grouped['XLE'] = 'X'
protein_letters_3to1_extended_grouped['PYL'] = 'K'
protein_letters_3to1_extended_grouped['SEC'] = 'C'


def filter_structure(model, ignore_disordered, ignore_hetatoms=True):
    """
    Filter a structure to remove HETATM or disordered atoms.

    Args:
        model (Bio.PDB.Model): model to filter
        ignore_disordered (bool): if True, remove disordered atoms
        ignore_hetatoms (bool): if True, remove HETATM that are not part of the protein

    Returns:
        None, the model is modified in place
    """

    if ignore_hetatoms:
        # removing HETATM that are not part of the protein
        for residue in list(model.get_residues()):
            if residue.get_id()[0] != " ":
                residue.get_parent().detach_child(residue.get_id())

    # remove disordered atoms
    if ignore_disordered:
        for residue in list(model.get_residues()):
            for atom in list(residue.get_atoms()):
                if atom.is_disordered() or atom.get_altloc() == "A" or atom.get_altloc() == "1":
                    residue.detach_child(atom.get_id())

        # remove residues with no atoms (if they appear after removing disordered atoms)
        for residue in list(model.get_residues()):
            if len(residue.get_unpacked_list()) == 0:
                residue.get_parent().detach_child(residue.get_id())


def residues_depth(model, surface_vertexes, ignore_disordered=False, sort_chains=False):
    """
    Calculate the depth of each residue in the model. The depth is the distance from the closest point on the surface.

    Args:
        model (Bio.PDB.Model): model to calculate the depth
        surface_vertexes (list): list of surface vertexes
        ignore_disordered (bool): if True, remove disordered atoms

    Returns:
        dict: dictionary with the depth of each residue, the key is a tuple (residue_id, chain_id, res_1_let_name)
    """

    depth_dict = {}
    filter_structure(model, ignore_disordered)

    chains = Selection.unfold_entities(model, "C")
    if sort_chains:
        chains = sorted(chains, key=lambda c: c.id)

    # calculate rdepth for each residue
    for chain in chains:
        for residue in chain:
            rd = residue_depth(residue, surface_vertexes)
            ca_rd = ca_depth(residue, surface_vertexes)
            res_1_let_name = protein_letters_3to1_extended_grouped.get(residue.get_resname(), "X")

            # TODO: some ca_rd are nan (because the residue has no CA atom), we set them to 0
            if ca_rd is None:
                ca_rd = 0

            depth_dict[(residue.get_id(), residue.get_parent().get_id(), res_1_let_name)] = (rd, ca_rd)

    return depth_dict


def get_seq_from_struct(model, ignore_disordered=False, sort_chains=False):
    filter_structure(model, ignore_disordered=ignore_disordered)
    seq_from_struct = []

    # we make sure we sort chains by id in the case of multichain proteins
    # For Scope database we should not sort them as it breaks the connection to the seq records
    # e.g 1b35 D:,C has the D chain first in the pdb file and D chain first in the seq record
    chains = Selection.unfold_entities(model, "C")
    if sort_chains:
        chains = sorted(chains, key=lambda c: c.id)

    for chain in chains:
        for residue in chain:
            res_1_let_name = protein_letters_3to1_extended_grouped.get(residue.get_resname(), "X")
            seq_from_struct.append(res_1_let_name)

    return seq_from_struct


def residue_solvent_accessible_surface_area(model, ignore_disordered=False, sort_chains=False):
    """
    Calculate the solvent accessible surface area for each residue in the model.
    It uses the Shrake Rupley algorithm from biopython. Some of the atoms are missing from the biopython source code
    and we add them here.

    Args:
        model (Bio.PDB.Model): model to calculate the solvent accessible surface area
        ignore_disordered (bool): if True, remove disordered atoms

    Returns:
        dict: dictionary with the solvent accessible surface area of each residue, the key is a tuple (residue_id, chain_id, res_1_let_name)

    """

    filter_structure(model, ignore_disordered)
    # deuterium (D) atom is heavy hidrogen and its missing from the biopython source code
    # we ignore the unknown atoms (X) because there are only 82 in the whole scop 2.06
    sr = ShrakeRupley(radii_dict={"D": 1.20, "X": 0.01})
    sr.compute(model, level="R")
    rsasa = {}

    chains = Selection.unfold_entities(model, "C")
    if sort_chains:
        chains = sorted(chains, key=lambda c: c.id)

    for chain in chains:
        for residue in chain:
            res_1_let_name = protein_letters_3to1_extended_grouped.get(residue.get_resname(), "X")
            rsasa[(residue.get_id(), residue.get_parent().get_id(), res_1_let_name)] = residue.sasa
    return rsasa


def get_alignment_mask(seq, seq_extracted_from_struct):
    """
    It aligns the true protein sequence with the sequence extracted from the structure.
    Some of the AA might be missing in the structure, so we need to find their position in the protein sequence.
    The alignment is done with the globalxs algorithm from biopython. We don't care about the scores,
    we just need the alignment.

    Args:
        seq (str): true protein sequence
        seq_extracted_from_struct (str): sequence extracted from the structure

    Returns:
        list: list of booleans, True if the AA is in the structure, False otherwise
    """

    # we do not need to mask if the seq extracted from structure has the same length
    # even if there are some AA in the structure that don't have the right name (e.g. X unknown )
    if len(seq) == len(seq_extracted_from_struct):
        return [True for _ in range(len(seq))]

    # should not happen with scop domains
    if len(seq) < len(seq_extracted_from_struct):
        print("Seq coming from structure is longer!")
        print(seq, "seq")
        print(seq_extracted_from_struct, "seq_extracted_from_struct")

    # CFMYGSK  and CFXSK can be aligned as CF---XSK, CFX--SK, CF-X-SK, CF--XSK
    # globalxs: penalises opening and extending gaps in both sequences the same
    # the alignment should have the same length as the protein seq
    # -1 for opening gap, -100 for extending gap to make sure we don't have large gaps
    alignments = pairwise2.align.globalxs(seq, seq_extracted_from_struct, -100, -1, one_alignment_only=True)
    # getting the protein seq part of the alignment
    alignment = alignments[0][1]

    # True where we have an AA at the structure matching the seq one, false otherwise
    mask = np.where(np.array(list(alignment)) != '-', True, False)

    if not len(mask) == len(alignment) == len(seq):
        print("Mask: ", mask)
        print("Alignment: ", alignment)
        print("Seq: ", seq)
        print("Seq extracted from struct: ", seq_extracted_from_struct)

    assert len(mask) == len(alignment) == len(seq)
    return list(mask)

def remove_chains_from_model(model, chains_to_remove):
    """
    Remove chains from a model

    Args:
        model (Bio.PDB.Model): model to remove chains from
        chains_to_remove (list): list of chain ids to remove

    Returns:
        None, it modifies the model in place
    """

    original_nr_chains = len(Selection.unfold_entities(model, "C"))
    for chain in Selection.unfold_entities(model, "C"):
        if chain.id in chains_to_remove:
            model.detach_child(chain.get_id())

    assert original_nr_chains - len(chains_to_remove) == len(Selection.unfold_entities(model, "C"))


# removing alternating positions of atoms (occupancy atoms)
class NotDisordered(Select):
    """
    Selects all atoms that are not disordered or are disordered and have altloc "A" or "1"
    To be used with io.save filtering
    """

    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A" or atom.get_altloc() == "1"


def expand_struct_data_to_full_seq(aa_mask, data):
    """
    Expands a sequence from the structure to the full sequence. The full sequence is the sequence of the protein

    Args:
        aa_mask (np.array): mask of the amino acids that appear in the structure
        data (np.array): data to fill in positions that appear in the structure

    """
    data_expanded = np.nan * np.ones(len(aa_mask))
    data_expanded[aa_mask] = data
    return data_expanded
