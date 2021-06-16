import os
import sys
import time
import tqdm
import argparse
import numpy as np
import pickle as pkl
import multiprocessing
from collections import defaultdict, Counter
from glob import glob
from functools import partial
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import obabel_utils
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from ogb.lsc import PCQM4MDataset
import pandas as pd
import pgl

from extended_feature import smiles2graph,mol2graph

MST_MAX_WEIGHT = 100 

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=False)

def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=False)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) #We assume this is not None
    return new_mol

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    return mol

def smiles_to_moltree(smiles):
    # cliques list of list
    # edges: edges between clique
    mol = get_mol(smiles)
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    cliques.extend(ssr)

    # which clique the atom belongs 
    nei_list = [[] for i in range(n_atoms)]
    for i, clique in enumerate(cliques):
        for atom in clique:
            nei_list[atom].append(i)

    #Build edges and add singleton cliques
    edges = defaultdict(int)

    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]

        bonds = [c for c in cnei if len(cliques[c]) == 2]
        rings = [c for c in cnei if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): #In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = 1
        elif len(rings) > 2: #Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1,c2)] = MST_MAX_WEIGHT - 1
        else:
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):
                        edges[(c1,c2)] = len(inter) #cnei[i] < cnei[j] by construction

    edges = [ (c1, c2, MST_MAX_WEIGHT-v,) for (c1, c2), v in edges.items()]
    if len(edges) == 0:
        return cliques, edges

    row, col, data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i],col[i]) for i in range(len(row))]
    return (cliques, edges)

def compute_3d_graph(input_s):
    mol = get_mol(input_s)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True,useBasicKnowledge=True)
    AllChem.UFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    coord = []
    for atom in mol.GetAtoms():
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        coord.append([pos.x, pos.y, pos.z])
    return coord
    

def smiles2graph_and_junction_tree(smiles_string, y):
    #cliques, edges = smiles_to_moltree(smiles_string)
    graph = smiles2graph(smiles_string)
    #g_dict = dict()
    #g_dict["mol_graph"] = graph
    #g_dict["junction_tree"] = dict()
    #g_dict["junction_tree"]["num_nodes"] = len(cliques) 
    #g_dict["junction_tree"]["edge_index"] = np.array(edges, dtype="int64").T   
    #g_dict["junction_tree"]["junc_dict"] = [] 
    #g_dict["mol2juct"] = []   
    #for cli_id, clique in enumerate(cliques): 
    #    for atom in clique:
    #        g_dict["mol2juct"].append((atom, cli_id))
    #g_dict["mol2juct"] = np.array(g_dict["mol2juct"], dtype="int64")

    #  try:
    #      coord = compute_3d_graph(smiles_string)
    #      graph["mol_coord"] = np.array(coord, dtype="float32")
    #  except:
    #      graph["mol_coord"] = np.zeros([graph["num_nodes"], 3], dtype="float32")
    new_graph = {}
    new_graph["smiles"] = smiles_string
    new_graph["edges"] = graph["edge_index"].T
    new_graph["num_nodes"] = graph["num_nodes"]
    new_graph["node_feat"] = graph["node_feat"]
    new_graph["node_feat_float"] = np.array(graph["node_feat_float"], dtype="float32")
    new_graph["edge_feat"] = graph["edge_feat"]
    #  new_graph["mol_coord"] = graph["mol_coord"]
    new_graph["label"] = y 
    #  if new_graph["mol_coord"].shape[0] != new_graph["num_nodes"]:
    #      graph["mol_coord"] = np.zeros([graph["num_nodes"], 3], dtype="float32")
    return new_graph 


def smile_func(mol_homo):
    return mol2data(mol_homo)

# def mol2data(smiles, mol, y, khop=1):
#     graph = mol2graph(smiles, mol, khop=khop)
#     new_graph = {}
#     new_graph["smiles"] = smiles
#     new_graph["edges"] = graph["edge_index"].T
#     new_graph["num_nodes"] = graph["num_nodes"]
#     new_graph["node_feat"] = graph["node_feat"]
#     new_graph["node_feat_float"] = np.array(graph["node_feat_float"], dtype="float32")
#     new_graph["edge_feat"] = graph["edge_feat"]
#     #  new_graph["mol_coord"] = graph["mol_coord"]
#     new_graph["label"] = y 
#     #  if new_graph["mol_coord"].shape[0] != new_graph["num_nodes"]:
#     #      graph["mol_coord"] = np.zeros([graph["num_nodes"], 3], dtype="float32")
#     return new_graph, graph

def mol2data(mol_homo, khop=1):
    i, smiles, homolumogap, mol = mol_homo
    if i % 1000 == 0:
        print(f"Converting {i}th mol...", flush=True)

    graph = mol2graph(smiles, mol, khop=khop)

    assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert(len(graph['node_feat']) == graph['num_nodes'])
    new_graph = {}
    new_graph["smiles"] = smiles
    new_graph["edges"] = graph["edge_index"].T
    new_graph["num_nodes"] = graph["num_nodes"]
    new_graph["node_feat"] = graph["node_feat"]
    new_graph["node_feat_float"] = np.array(graph["node_feat_float"], dtype="float32")
    new_graph["edge_feat"] = graph["edge_feat"]
    #  new_graph["mol_coord"] = graph["mol_coord"]
    new_graph["label"] = homolumogap 
    #  if new_graph["mol_coord"].shape[0] != new_graph["num_nodes"]:
    #      graph["mol_coord"] = np.zeros([graph["num_nodes"], 3], dtype="float32")
    return new_graph, graph


def read_sdf_files(file, from_dft=False):
    sdf_list = []
    with open(file) as ob_f:
        for line in ob_f:
            segs = line.strip('\n').split('\t')
            if len(segs) == 4:
                idx, smiles, pcqm4m_gap, flat_sdf_string = segs
                sdf_string = obabel_utils.deflatten_sdf_string(flat_sdf_string)
                try:
                    mol = obabel_utils.get_mol_withH_from_sdf_string(sdf_string)
                    if from_dft:
                        if not mol.HasProp('cal_homolumogap'):
                            mol = None
                            break
                        pcqm4m_gap = float(pcqm4m_gap) if pcqm4m_gap != "" else 0
                        cal_gap = float(mol.GetProp('cal_homolumogap'))
                        if np.abs(pcqm4m_gap - cal_gap) > 0.08:  # if difference is too big, drop
                            mol = None
                except Exception as e:
                    print('[ERROR] read_sdf_files', e)
                    mol = None
            else:
                idx, smiles, pcqm4m_gap = segs
                mol = None
            pcqm4m_gap = float(pcqm4m_gap) if pcqm4m_gap != "" else 0
            sdf_list.append((int(idx), smiles, pcqm4m_gap, mol))
    return sdf_list
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data')
    parser.add_argument("--data_path", type=str, default="../dataset")
    parser.add_argument("--debug",action='store_true')
    args = parser.parse_args()

    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

#     dataset = PCQM4MDataset(root = args.data_path, only_smiles = True)
    max_workers = 32
    graph_list = []
    aux_data_list = []
    labels = []
    start = time.time()
    # Loading sdf data
    sdf_path = "../dataset/flat_pyscf_sdf"
    print('load sdf data from: %s' % sdf_path)
    file_list = sorted(glob('%s/*/*' % sdf_path))
    csv_path = os.path.join(args.data_path,"pcqm4m_kddcup2021","raw","data.csv.gz")
    data_df = pd.read_csv(csv_path)
    with multiprocessing.Pool(max_workers) as p:
        sdf_list = p.map(partial(read_sdf_files, from_dft=False), file_list)
    sdf_list = [item for sublist in sdf_list for item in sublist]     # flatten

    # some compounds may lost, get them back from origin data.csv
    
    
    filled_sdf_list = [0] * len(data_df)

    for item in sdf_list:
        filled_sdf_list[item[0]] = item    # item[0] is index
    for i, item in enumerate(filled_sdf_list):
        if item == 0:                      # 0 means lost
            filled_sdf_list[i] = list(data_df.iloc[i])+[None]
            
    if args.debug: 
        filled_sdf_list = filled_sdf_list[:12_000]
    print(f'number of molecules: {len(filled_sdf_list)}')
    graph_list = []
    aux_data_list = []
    khop = 1
    with multiprocessing.Pool(max_workers) as p:
        graph_homolumo_list = p.map(partial(mol2data, khop=khop), filled_sdf_list)
        
    for gdata, aux_data in graph_homolumo_list:
        g = pgl.Graph(
                edges=gdata['edges'],
                num_nodes=gdata['num_nodes'],
                node_feat={
                    'feat': gdata['node_feat'],
                    'feat_float': gdata['node_feat_float'],
                    },
                edge_feat={'feat': gdata['edge_feat']})
        graph_list.append(g)
        labels.append(gdata['label'])
        aux_data_list.append(aux_data)


    graph_list = pgl.Graph.batch(graph_list)
    processed_path = os.path.join(args.data_path, "aux_processed_data")
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    graph_list.dump(os.path.join(processed_path, "mmap_graph"))
    np.save(os.path.join(processed_path, "label.npy"), np.array(labels, dtype="float32")) 
    pkl.dump(aux_data_list, open(os.path.join(processed_path, "mol_tree_aux.pkl"),"wb"))
    print("total data process time: %s" % (time.time() -  start))

