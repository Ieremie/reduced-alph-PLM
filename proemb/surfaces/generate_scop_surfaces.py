import argparse
import glob
import os
import sys
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from subprocess import Popen, PIPE

from Bio.PDB.ResidueDepth import _get_atom_radius
import numpy as np
import pymesh
import tqdm
from Bio.PDB import PDBParser, Select, PDBIO, Selection, PDBList
from numpy.linalg import norm
from numpy.matlib import repmat
import pprint

import warnings

from sklearn.neighbors import KDTree

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from proemb.utils.surface_util import remove_chains_from_model, NotDisordered

warnings.filterwarnings("error")

'''
Total X atoms:  82
Total unknown radii X atoms:  82
Total HETATM atoms:  130196
Total unknown radii HETATM atoms:  64
Total disordered atoms:  384486
Total unknown radii disordered atoms:  3

It makes sense to ignore X atoms (set them to 0.01 RADII) as there are only a few

1. Surfaces without HETATM and with disordered atoms (choosing 1 of their location)
2. Surfaces without HETATM and without disordered atoms
3. Optional if needed we can calculate surfaces with HETATM, but these are not part of the protein 
so it can be difficult to interpret downstream experiments

All atoms with unknown radii are set to 0.01 (only a few in all SCOP 2.06)
'''

# radii for atoms in explicit case.
# TODO: The ones used here are not the same as in A. Bondi (1964). "van der Waals Volumes and Radii".
# TODO: In the dmasif code they seem to use the correct ones
# TODO: This is a crude version of RADII values as it maps all types of atoms begining with the same letter to the same radius
# TODO: The original pdb2xyzrn uses a more complete version of this (also biopython)
RADII = {"N": "1.540000", "O": "1.400000", "C": "1.740000", "H": "1.200000", "S": "1.800000", "P": "1.800000",
         "Z": "1.39", "X": "0.770000"}

# This  polar hydrogen's names correspond to that of the program Reduce.
polarHydrogens = {"ALA": ["H"], "GLY": ["H"], "SER": ["H", "HG"], "THR": ["H", "HG1"], "LEU": ["H"], "ILE": ["H"],
                  "VAL": ["H"], "ASN": ["H", "HD21", "HD22"], "GLN": ["H", "HE21", "HE22"],
                  "ARG": ["H", "HH11", "HH12", "HH21", "HH22", "HE"], "HIS": ["H", "HD1", "HE2"], "TRP": ["H", "HE1"],
                  "PHE": ["H"], "TYR": ["H", "HH"], "GLU": ["H"], "ASP": ["H"], "LYS": ["H", "HZ1", "HZ2", "HZ3"],
                  "PRO": [], "CYS": ["H"], "MET": ["H"]}
MSMS_BIN = os.environ['MSMS_BIN'] if 'MSMS_BIN' in os.environ else "/Applications/msms/msms.x86_64Darwin.2.6.1"

MESH_RES = 1
EPSILON = 1.0e-6

"""
protonate.py: Wrapper method for the reduce program: protonate (i.e., add hydrogens) a pdb using reduce 
                and save to an output file.
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0

protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
Remove protons first, in case the structure is already protonated
"""


def protonate(in_pdb_file, out_pdb_file):
    args = ["reduce", "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()
    # Now add them again.
    args = ["reduce", "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()


"""
msms_output_2numpy.py: Read an msms output file that was output by MSMS (MSMS is the program we use to build a surface) 
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0

Read the surface from the msms output. MSMS outputs two files: {file_root}.vert and {file_root}.face
"""


def msms_output_2numpy(file_base, biopython_version):
    vertfile = open(file_base + ".vert")
    meshdata = (vertfile.read().rstrip()).split("\n")
    vertfile.close()

    # Read number of vertices.
    count = {}

    # MSMS failed
    if len(meshdata) < 3:
        print("MSMS output is empty.")
        raise OSError()

    header = meshdata[2].split()
    count["vertices"] = int(header[0])
    ## Data Structures
    vertices = np.zeros((count["vertices"], 3))
    normalv = np.zeros((count["vertices"], 3))
    atom_id = [""] * count["vertices"]
    res_id = [""] * count["vertices"]
    for i in range(3, len(meshdata)):
        fields = meshdata[i].split()

        if len(fields) <= 8:
            print("MSMS output doesnt have the expected format.")
            raise OSError()
        vi = i - 3

        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        normalv[vi][0] = float(fields[3])
        normalv[vi][1] = float(fields[4])
        normalv[vi][2] = float(fields[5])
        # closest atom sphere, not used
        atom_id[vi] = fields[7]

        # the full id of the atom
        # If provided to MSMS as an extra column it would appear hese as output
        res_id[vi] = "" if biopython_version else fields[9]
        count["vertices"] -= 1

    # Read faces.
    facefile = open(file_base + ".face")
    meshdata = (facefile.read().rstrip()).split("\n")
    facefile.close()

    # Read number of vertices.
    header = meshdata[2].split()
    count["faces"] = int(header[0])
    faces = np.zeros((count["faces"], 3), dtype=int)

    for i in range(3, len(meshdata)):
        fi = i - 3
        fields = meshdata[i].split()
        # -1 probably to zero index the atom indices?
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1
        count["faces"] -= 1

    assert count["vertices"] == 0
    assert count["faces"] == 0
    # normalv is the normal of the vertices not of the faces
    return vertices, faces, normalv, res_id


def pdb_to_xyzr(pdbfilename, xyzrfilename, ignore_HETATM=True):
    """
    Using the implementation from Biopthon that uses the mode detailed atom radii values instead of clustered version
    as in MASIF implementation
    """

    parser = PDBParser(QUIET=True)
    model = parser.get_structure(pdbfilename, pdbfilename)

    # remove HETATM resiues from the model which are not part of the protein
    for residue in list(model.get_residues()):
        if residue.get_id()[0] != " " and ignore_HETATM:
            residue.get_parent().detach_child(residue.get_id())

    atom_list = Selection.unfold_entities(model, "A")
    outfile = open(xyzrfilename, "w")
    for atom in atom_list:
        x, y, z = atom.coord
        # use the detailed atom radii values

        # catch warning as if it was an error because it stops the process running for some reason
        try:
            radius = _get_atom_radius(atom, rtype="united")
        except:
            # print the warning
            print("Warning: ", sys.exc_info()[1])
            radius = 0.01  # default value if the radius is not found (same as in Biopython)

        outfile.write(f"{x:6.3f}\t{y:6.3f}\t{z:6.3f}\t{radius:1.2f}\n")

    # close the file
    outfile.close()


"""
xyzrn.py: Read a pdb file and output it is in xyzrn for use in MSMS
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

''' it creates a file with the 3D coordinates of each atom and the wan der wall radii
it also adds and id to each atom which is probably unique, the color im not sure what it is for
However theses are not used in the msms program'''


def pdb_as_xyzrn(pdbfilename, xyzrnfilename, ignore_HETATM):
    """
        pdbfilename: input pdb filename
        xyzrnfilename: output in xyzrn format.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdbfilename, pdbfilename)
    outfile = open(xyzrnfilename, "w")
    for atom in struct.get_atoms():
        atom_name = atom.get_name()
        residue = atom.get_parent()
        # Ignore hetatms.
        if residue.get_id()[0] != " " and ignore_HETATM:
            continue
        residue_name = residue.get_resname()
        chain_id = residue.get_parent().get_id()
        atom_type = atom_name[0]

        color = "Green"
        coords = None
        if atom_type in RADII and residue_name in polarHydrogens:
            if atom_type == "O":
                color = "Red"
            if atom_type == "N":
                color = "Blue"
            if atom_type == "H":
                if atom_name in polarHydrogens[residue_name]:
                    color = "Blue"  # Polar hydrogens
            coords = "{:.06f} {:.06f} {:.06f}".format(atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2])
            insertion = residue.get_id()[2] if residue.get_id()[2] != " " else "x"
            full_id = "{}_{:d}_{}_{}_{}_{}".format(chain_id, residue.get_id()[1], insertion, residue_name, atom_name,
                                                   color)

        if coords is not None:
            outfile.write(coords + " " + RADII[atom_type] + " 1 " + full_id + "\n")
    outfile.close()


'''
Pablo Gainza LPDI EPFL 2017-2019
Calls MSMS and returns the vertices.
Special atoms are atoms with a reduced radius.
'''


def computeMSMS(file_base, probe_radius):
    # Now run MSMS on xyzrn file

    # use 1.4 to be the same as the SASA computation in biopython
    args = [MSMS_BIN, "-density", "3.0", "-hdensity", "3.0", "-probe",
            f"{probe_radius}", "-if", file_base + ".xyzrn", "-of", file_base, "-af", file_base]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()


"""
fixmesh.py: Regularize a protein surface mesh. 
- based on code from the PyMESH documentation. 
"""


def fix_mesh(mesh, resolution, detail="normal"):
    bbox_min, bbox_max = mesh.bbox;
    diag_len = norm(bbox_max - bbox_min);
    if detail == "normal":
        target_len = diag_len * 5e-3;
    elif detail == "high":
        target_len = diag_len * 2.5e-3;
    elif detail == "low":
        target_len = diag_len * 1e-2;

    target_len = resolution
    # print("Target resolution: {} mm".format(target_len));
    # PGC 2017: Remove duplicated vertices first
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)

    count = 0;
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100);
    mesh, __ = pymesh.split_long_edges(mesh, target_len);
    num_vertices = mesh.num_vertices;
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6);
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                                               preserve_feature=True);
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100);
        if mesh.num_vertices == num_vertices:
            break;

        num_vertices = mesh.num_vertices;
        # print("#v: {}".format(num_vertices));
        count += 1;
        if count > 10: break;

    mesh = pymesh.resolve_self_intersection(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh = pymesh.compute_outer_hull(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5);
    mesh, __ = pymesh.remove_isolated_vertices(mesh);
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)

    return mesh


"""
compute_normal.py: Compute the normals of a closed shape.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF, based on previous matlab code by Gabriel Peyre, converted to Python by Pablo Gainza
"""


def compute_normal(vertex, face):
    """
    compute_normal - compute the normal of a triangulation
    vertex: 3xn matrix of vertices
    face: 3xm matrix of face indices.

      normal,normalf = compute_normal(vertex,face)

      normal(i,:) is the normal at vertex i.
      normalf(j,:) is the normal at face j.

    Copyright (c) 2004 Gabriel Peyr
    Converted to Python by Pablo Gainza LPDI EPFL 2017
    """

    vertex = vertex.T
    face = face.T
    nface = np.size(face, 1)
    nvert = np.size(vertex, 1)
    normal = np.zeros((3, nvert))
    # unit normals to the faces
    normalf = crossp(
        vertex[:, face[1, :]] - vertex[:, face[0, :]],
        vertex[:, face[2, :]] - vertex[:, face[0, :]],
    )
    sum_squares = np.sum(normalf ** 2, 0)
    d = np.sqrt(sum_squares)
    d[d < EPSILON] = 1
    normalf = normalf / repmat(d, 3, 1)
    # unit normal to the vertex
    normal = np.zeros((3, nvert))
    for i in np.arange(0, nface):
        f = face[:, i]
        for j in np.arange(3):
            normal[:, f[j]] = normal[:, f[j]] + normalf[:, i]

    # normalize
    d = np.sqrt(np.sum(normal ** 2, 0))
    d[d < EPSILON] = 1
    normal = normal / repmat(d, 3, 1)
    # enforce that the normal are outward
    vertex_means = np.mean(vertex, 0)
    v = vertex - repmat(vertex_means, 3, 1)
    s = np.sum(np.multiply(v, normal), 1)
    if np.sum(s > 0) < np.sum(s < 0):
        # flip
        normal = -normal
        normalf = -normalf
    return normal.T


def crossp(x, y):
    # x and y are (m,3) dimensional
    z = np.zeros((x.shape))
    z[0, :] = np.multiply(x[1, :], y[2, :]) - np.multiply(x[2, :], y[1, :])
    z[1, :] = np.multiply(x[2, :], y[0, :]) - np.multiply(x[0, :], y[2, :])
    z[2, :] = np.multiply(x[0, :], y[1, :]) - np.multiply(x[1, :], y[0, :])
    return z


def save_ply(filename, vertices, faces=[], normals=None, iface=None):
    """ Save vertices, mesh in ply format.
        vertices: coordinates of vertices
        faces: mesh

    read_ply.py: Save a ply file to disk using pymesh and load the attributes used by MaSIF.
    Pablo Gainza - LPDI STI EPFL 2019
    Released under an Apache License 2.0
    """
    mesh = pymesh.form_mesh(vertices, faces)
    if normals is not None:
        n1 = normals[:, 0]
        n2 = normals[:, 1]
        n3 = normals[:, 2]
        mesh.add_attribute("vertex_nx")
        mesh.set_attribute("vertex_nx", n1)
        mesh.add_attribute("vertex_ny")
        mesh.set_attribute("vertex_ny", n2)
        mesh.add_attribute("vertex_nz")
        mesh.set_attribute("vertex_nz", n3)

    if iface is not None:
        mesh.add_attribute("vertex_iface")
        mesh.set_attribute("vertex_iface", iface)

    pymesh.save_mesh(filename, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)


def generate_surface(pdb_path, args, complex_mesh_pth=None, selected_chains=None):
    # ------------- select only the first model present in the PDB file ---------------
    model = PDBParser(QUIET=True).get_structure(pdb_path, pdb_path)[0]

    if selected_chains is not None:
        all_chains = Selection.unfold_entities(model, "C")
        chains_to_remove = [chain.id for chain in all_chains if chain.id not in selected_chains]
        remove_chains_from_model(model, chains_to_remove)

    io = PDBIO()
    io.set_structure(model)

    if args['ignore_disordered']:
        io.save(pdb_path + "model", select=NotDisordered())
    else:
        io.save(pdb_path + "model")

    # ---------------------adding hydrogen atoms to the protein------------------------
    protonate(pdb_path + "model", pdb_path + 'protonated')

    # atom-average folder for generated output files with unique id (no multiprocess crashes)
    file_base = tempfile.gettempdir() + '/id_' + str(uuid.uuid4())

    # ------------ generating the file needed for MSMS input in atom-average directory---------
    if not args['biopython_version']:
        pdb_as_xyzrn(pdb_path + 'protonated', file_base + ".xyzrn", args['ignore_HETATM'])
    else:
        pdb_to_xyzr(pdb_path + 'protonated', file_base + ".xyzrn", args['ignore_HETATM'])

    try:
        # ------------ generating the surface------------------------------------------
        computeMSMS(file_base, args['probe_radius'])
    except OSError as e:
        print('.xyzrn NOT generated for: ', pdb_path, ", chain:", selected_chains)

    chain_names = '_' + ''.join(selected_chains) if selected_chains is not None else ''
    base_path = os.path.dirname(pdb_path)
    pdb_name = os.path.basename(pdb_path)[:-4]
    try:
        vertices, faces, normals, _ = msms_output_2numpy(file_base, args['biopython_version'])

        # ------------ regularize the mesh and recalculate the normals, save it--------
        mesh = pymesh.form_mesh(vertices, faces)
        regular_mesh = fix_mesh(mesh, MESH_RES)
        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

        # -----------------saving the surface as a ply file----------------------------
        if complex_mesh_pth is not None:
            iface = compute_iface(complex_mesh_pth=complex_mesh_pth, regular_mesh=regular_mesh)
        else:
            iface = None

        save_ply(f'{base_path}/ply/{pdb_name}{chain_names}.ply', regular_mesh.vertices,
                 regular_mesh.faces, normals=vertex_normal, iface=iface)


    except OSError as e:
        print('MSMS output NOT generated for: ', pdb_path, ", chain:", selected_chains)
    except RuntimeWarning as warning:
        print(warning, '/n', ' for: ', pdb_path, ", chain:", selected_chains)

    try:
        # ----------------------Remove temporary files---------------------------------
        os.remove(file_base + '.area')
        os.remove(file_base + '.vert')
        os.remove(file_base + '.face')
        os.remove(file_base + '.xyzrn')
        os.remove(pdb_path + 'protonated')
        os.remove(pdb_path + 'model')
    except OSError:
        pass

    if not Path(f'{base_path}/ply/{pdb_name}{chain_names}.ply').is_file():
        print(".ply NOT generated for: ", pdb_path)

def compute_iface(complex_mesh_pth, regular_mesh):
    """
    Compute the interface between the regular_mesh and the complex_mesh.
    Taken from the MASIF code and adapted.

    Args:
        complex_mesh_pth (str): Path to the mesg of the complex.
        regular_mesh (pymesh.mesh.Mesh): The mesh on which we compute the interface.

    Returns:
        iface (np.array): Array of 0 and 1, where 1 indicates that the vertex is on the interface.
    """

    complex_mesh = pymesh.load_mesh(complex_mesh_pth)
    complex_mesh = pymesh.form_mesh(complex_mesh.vertices, complex_mesh.faces)

    # try to fix the mesh, if it fails, it is degenerated.
    # the mesh was initially fixed upon generation, so it should be fine.
    '''
    try:
        complex_mesh = fix_mesh(complex_mesh, MESH_RES)
    except RuntimeError as e:
        print(e)
        print("Re-fixing the complex mesh FAILED: ", complex_mesh_pth)
    '''

    iface = np.zeros(len(regular_mesh.vertices))

    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = KDTree(complex_mesh.vertices)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d)  # Square d, because this is how it was in the pyflann version.
    assert (len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0

    return iface


def download_pdb(pdb_ids, pdb_dir):
    """
    Download pdb files from the PDB database.

    Args:
        pdb_ids (list): list of pdb ids
        pdb_dir (str): path to the directory where the pdb files will be saved

    Returns:
        None, saves the pdb files in the pdb_dir directory
    """

    # ----------------------------- download pdb files -------------------- -----------------
    for group_id in tqdm.tqdm(pdb_ids, total=len(pdb_ids)):

        complex_name = f'pdb{group_id}.ent'
        if os.path.exists(pdb_dir + complex_name):
            continue

        pdbl = PDBList(server='http://ftp.wwpdb.org', verbose=False)
        pdbl.retrieve_pdb_file(group_id, pdir=pdb_dir, file_format='pdb')

        if not os.path.exists(pdb_dir + complex_name):
            print("Trying obsolete PDBs...")
            pdbl = PDBList(server='http://ftp.wwpdb.org', verbose=False, obsolete_pdb=pdb_dir)
            pdbl.retrieve_pdb_file(group_id, pdir=pdb_dir, file_format='pdb', obsolete=True)

        if not os.path.exists(pdb_dir + complex_name):
            print("PDB file not found: ", pdb_dir + complex_name)
            continue


def generate_surfaces_scop(args):
    # we want to see warnings related to degenerated surfaces(complex loops)
    warnings.filterwarnings("error")
    scop_pdb_paths = sorted(glob.glob(f'{args["root"]}{os.sep}*{os.sep}*.ent'))

    # if we are running multiple parallel processes
    # each process works on a separate (almost) equal chunk of data
    if args['multi_process']:
        # make sure processes don't work in the same subfolder
        scop_pdb_paths = sorted(glob.glob(f'{args["root"]}{os.sep}*'))
        scop_pdb_paths = np.array_split(scop_pdb_paths, int(os.environ["SLURM_NTASKS_PER_NODE"]) *
                                        int(os.environ["SLURM_NNODES"]))
        scop_pdb_paths = scop_pdb_paths[int(os.environ["SLURM_PROCID"])]
        scop_pdb_paths = [file for folder in scop_pdb_paths for file in glob.glob(f'{folder}{os.sep}*.ent')]

    for scop_pdb_path in tqdm.tqdm(scop_pdb_paths):
        generate_surface(scop_pdb_path, args)



def generate_surfaces_ppi(ppi_file='../../data/masif-ppis/lists/all_ppi.txt',
                          pdb_dir='../../data/masif-ppis/pdb/',
                          args=None):
    """
    Generate surfaces for all PPIs in the list.
    Input:
        ppi_file: path to the file containing the list of PPI ids.
        pdb_dir: path to the directory containing the PDB files.
        args: arguments for the surface generation.
    Output:
        None.
    """

    with open(ppi_file, 'r') as f:
        ppi_list = f.readlines()

    ppi_list = [l.strip('\n').lower() for l in ppi_list]
    # dict of PDBid -> [pdbid_chainsA, pdbid_chainsB, ...]
    grouped_pdbs = defaultdict(list)
    # all the chains in the PDBs that are part of a protein interaction
    # dict of PDBid -> set([chain1, chain2, ...])
    complex_chains = defaultdict(list)
    for ppi_id in ppi_list:
        pdb_id = ppi_id.split('_')[0]
        grouped_pdbs[pdb_id].append(ppi_id)

        complex_chains[pdb_id] = sorted(list(
            set([chain.upper() for chains in ppi_id.split("_")[1:] for chain in chains] + complex_chains[pdb_id])))

    # split the complex ids between processes to avoid file clashes
    if args['multi_process']:
        split_keys = np.array_split(list(grouped_pdbs.keys()), int(os.environ["SLURM_NTASKS_PER_NODE"]) *
                                    int(os.environ["SLURM_NNODES"]))
        grouped_pdbs = {k: grouped_pdbs[k] for k in split_keys[int(os.environ["SLURM_PROCID"])]}

    # download pdb files for the whole complex
    download_pdb(list(grouped_pdbs.keys()), pdb_dir)

    # generate surfaces
    for pdb_id in tqdm.tqdm(grouped_pdbs.keys(), total=len(grouped_pdbs)):
        # first we generate the complex surface
        # we only include chains that are found in PPI we are generating surfaces for
        # this is to remove extra chains that are in the crystal structure but not in the PPI
        # For example: We have a complex with 4 chains, but only 2 unique chains are in the PPI.
        # This is because of a duplication in crystal unit.
        # Some of them might cover our chains of interest, and we will label interfaces wrongly

        # using all chains found in ppis
        selected_chains = complex_chains[pdb_id]
        generate_surface(f'{pdb_dir}pdb{pdb_id}.ent', args, complex_mesh_pth=None,
                         selected_chains=selected_chains)
        complex_mesh_path = f'{pdb_dir}ply/pdb{pdb_id}_{"".join(selected_chains)}.ply'

        for ppi_id in grouped_pdbs[pdb_id]:

            # surfaces for interacting parts
            # only the chains of the interacting part
            generate_surface(f'{pdb_dir}pdb{pdb_id}.ent', args, complex_mesh_pth=complex_mesh_path,
                             selected_chains=list(ppi_id.split("_")[1].upper()))
            # chains of the opposite interacting part
            generate_surface(f'{pdb_dir}pdb{pdb_id}.ent', args, complex_mesh_pth=complex_mesh_path,
                             selected_chains=list(ppi_id.split("_")[2].upper()))


def main():
    parser = argparse.ArgumentParser('Script for generating protein surfaces')
    parser.add_argument('--root', default='../../data/SCOPe/pdbstyle-2.06-structures-radius-1.4-no-HETATM')
    parser.add_argument('--multi_process', action='store_true', help='each process works on partition of the data')

    parser.add_argument('--biopython_version', action='store_true', help='use biopython version of pdb_to_xyzr')
    parser.add_argument('--ignore_HETATM', action='store_true', help='ignore HETATM')
    parser.add_argument('--probe_radius', type=float, default=1.4, help='probe radius for MSMS')
    parser.add_argument('--ignore_disordered', action='store_true', help='ignore disordered atoms')

    parser.add_argument('--ppi', action='store_true', help='generate surfaces for PPIs')
    parser.add_argument('--scop', action='store_true', help='generate surfaces for SCOPe')

    args = parser.parse_args()
    # args to dict
    args = vars(args)

    # print args only on one of the processes
    if not args['multi_process'] or int(os.environ["SLURM_PROCID"]) == 0:
        pprint.pprint(args)

    if args['ppi']:
        generate_surfaces_ppi(args=args, ppi_file=args['root'] + '/masif-ppis/lists/all_ppi.txt',
                              pdb_dir=args['root'] + '/masif-ppis/pdb/')
    elif args['scop']:
        generate_surfaces_scop(args)


if __name__ == '__main__':
    main()
