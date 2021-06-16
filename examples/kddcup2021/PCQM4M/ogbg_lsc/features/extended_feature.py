import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.rdPartialCharges

from ogb.utils.features import (allowable_features, atom_to_feature_vector, 
                                atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
from local_feature import bond_to_feature_vector, get_bond_feature_dims
from copy import deepcopy
import hashlib

def rdkit_embed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mol = Chem.RemoveHs(mol)
    return mol

def rdkit_embed2d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    return mol

def shortest_path(mol):
    matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    return matrix

def get_graph_str(graph, atom_index, nei_atom_indices, nei_bond_indices):
    """tbd"""
    atomic_num = graph['node_feat'][:, 0]
    bond_type = graph['edge_feat'][:, 0]
    subgraph_str = 'A' + str(atomic_num[atom_index])
    subgraph_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
    subgraph_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
    return subgraph_str

def get_graph_twohop_str(graph, atom_index,
                         nei_atom_indices, nei_bond_indices,
                         neinei_atom_indices, neinei_bond_indices):
    """tbd"""
    atomic_num = graph['node_feat'][:, 0]
    bond_type = graph['edge_feat'][:, 0]
    subgraph_twohop_str = 'A' + str(atomic_num[atom_index])
    subgraph_twohop_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
    subgraph_twohop_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
    subgraph_twohop_str += 'B' + ':'.join([str(x) for x in np.sort(atomic_num[neinei_atom_indices])])
    subgraph_twohop_str += 'S' + ':'.join([str(x) for x in np.sort(bond_type[neinei_bond_indices])])

    return subgraph_twohop_str

def get_graph_threehop_str(graph, atom_index, nei_atom_indices, nei_bond_indices,
                           neinei_atom_indices, neinei_bond_indices,
                           neineinei_atom_indices, neineinei_bond_indices):
    """tbd"""
    atomic_num = graph['node_feat'][:, 0]
    bond_type = graph['edge_feat'][:, 0]
    subgraph_threehop_str = 'A' + str(atomic_num[atom_index])
    subgraph_threehop_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
    subgraph_threehop_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
    subgraph_threehop_str += 'B' + ':'.join([str(x) for x in np.sort(atomic_num[neinei_atom_indices])])
    subgraph_threehop_str += 'S' + ':'.join([str(x) for x in np.sort(bond_type[neinei_bond_indices])])
    subgraph_threehop_str += 'P' + ':'.join([str(x) for x in np.sort(atomic_num[neineinei_atom_indices])])
    subgraph_threehop_str += 'Q' + ':'.join([str(x) for x in np.sort(bond_type[neineinei_bond_indices])])

    return subgraph_threehop_str

def md5_hash(string):
    """tbd"""
    md5 = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(md5, 16)

def mol2graph(smiles, mol, khop=1):
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        dft_success = 1
    except Exception as e:
        # mol = rdkit_embed(smiles)
        mol = rdkit_embed2d(smiles)
        # conf = mol.GetConformer()
        # print(conf.GetAtomPosition(0).x, conf.GetAtomPosition(0).y, conf.GetAtomPosition(0).z)
        dft_success = 0
    mol.UpdatePropertyCache()
    graph = rdkit_mol2graph(mol, khop, with_3d=True)
    graph["smiles"] = smiles
    graph['dft_success'] = dft_success
    return graph

def smiles2graph(smiles_string, khop=1):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    mol = Chem.MolFromSmiles(smiles_string)
    return rdkit_mol2graph(mol, khop)

def rdkit_mol2graph(mol, khop, with_3d=False):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))

    atom_features_float_list = [[] for i in range(len(atom_features_list))]
    # extra atom features
    ring_size = get_ring_size(mol)
    partial_charge = get_partial_charge(mol)
    valence = get_valence_of_out_shell(mol)
    van_der_waals_radis = get_van_der_waals_radis(mol)
    for i in range(len(mol.GetAtoms())):
        atom_features_list[i].extend(ring_size[i])
        atom_features_list[i].append(valence[i])

        atom_features_float_list[i].append(partial_charge[i])
        atom_features_float_list[i].append(van_der_waals_radis[i])

    x = np.array(atom_features_list, dtype = np.int64)
    x_float = np.array(atom_features_float_list, dtype = np.float)

    # bonds
    num_bond_features = 4  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) == 0: # mol has bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)
    else:
        # real edges
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # ###########
            # # DEBUG
            # conf = mol.GetConformer()
            # v1 = np.array([conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])
            # v2 = np.array([conf.GetAtomPosition(j).x, conf.GetAtomPosition(j).y, conf.GetAtomPosition(j).z])
            # print('bl', np.linalg.norm(v1 - v2))
            # ###########

            edge_feature = bond_to_feature_vector(bond)
            edge_feature.append(1)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['node_feat_float'] = x_float
    graph['num_nodes'] = len(x)
    graph['num_edges'] = edge_index.shape[1]

    graph = gen_context_id(graph)

    if with_3d:
        graph = parse_3dpos(graph, mol, khop, gen_aux_task=True)

    return graph

def gen_context_id(graph):
    g = graph
    N = graph['num_nodes']
    E = graph['num_edges']
    full_bond_indices = np.arange(E)

    target_labels = []
    target_twohop = []
    # target_threehop = []

    for atom_index in range(N):
        nei_bond_indices = full_bond_indices[g['edge_index'][0, :] == atom_index]
        nei_atom_indices = g['edge_index'][ 1, nei_bond_indices]

        neinei_bond_indices = full_bond_indices[np.isin(g['edge_index'][0, :], nei_atom_indices)]
        neinei_atom_indices = g['edge_index'][1, neinei_bond_indices]

        # neineinei_bond_indices = full_bond_indices[np.isin(g['edge_index'][0, :], neinei_atom_indices)]
        # neineinei_atom_indices = g['edge_index'][1, neineinei_bond_indices]

        subgraph_str = get_graph_str(g, atom_index, nei_atom_indices, nei_bond_indices)
        subgraph_id = md5_hash(subgraph_str) % 1000
        target_labels.append(subgraph_id)

        subgraph_twohop_str = get_graph_twohop_str(graph, atom_index, nei_atom_indices, nei_bond_indices, neinei_atom_indices, neinei_bond_indices)
        subgraph_twoid = md5_hash(subgraph_twohop_str) % 5000
        target_twohop.append(subgraph_twoid)

        # subgraph_threehop_str = get_graph_threehop_str(graph, atom_index, nei_atom_indices, nei_bond_indices, neinei_atom_indices, neinei_bond_indices, neineinei_atom_indices, neineinei_bond_indices)
        # subgraph_threeid = md5_hash(subgraph_threehop_str) % 5000
        # target_threehop.append(subgraph_threeid)

    target_labels = np.array(target_labels)
    graph['context_id'] = target_labels
    graph['twohop_context'] = target_twohop
    # graph['threehop_context'] = target_threehop
    return graph

def parse_3dpos(graph, mol, khop, rtn_atom_pos=False, bond_by_dis=False, gen_aux_task=False):
    # atom positions
    atom_position_list = [[] for i in range(mol.GetNumAtoms())]
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        atom_position_list[i].append(conf.GetAtomPosition(i).x)
        atom_position_list[i].append(conf.GetAtomPosition(i).y)
        atom_position_list[i].append(conf.GetAtomPosition(i).z)
    x_position = np.array(atom_position_list, dtype=np.float)

    if rtn_atom_pos:
        graph['node_position'] = x_position

    # edge
    matrix_3ddis = Chem.rdmolops.Get3DDistanceMatrix(mol)
    matrix_topodis = shortest_path(mol)
    edge_features_float_list = []

    # add distance for real bonds (hop == 1)
    if mol.GetNumBonds() > 0:
        for i in range(len(graph['edge_index'][0])):
            source_id, dest_id = graph['edge_index'][:, i]
            edge_features_float_list.append([matrix_3ddis[source_id][dest_id]])
        edge_attr_float = np.array(edge_features_float_list, dtype=np.float)
    else:
        num_bond_features_float = 1
        edge_attr_float = np.empty((0, num_bond_features_float), dtype=np.float)

    # add fully connected edges
    if khop > 1 or bond_by_dis:
        edges_list = []
        edge_features_list = []
        edge_features_float_list = []
        for i in range(len(matrix_topodis)):
            for j in range(len(matrix_topodis[0])):
                if matrix_topodis[i][j] > 1 and (matrix_topodis[i][j] <= khop or \
                    (bond_by_dis and matrix_3ddis[i][j] < bond_by_dis)):       # 1 hop has been appended before
                    # virtual feature for k-hop connected bond, value = len()-1
                    edge_feature = [i - 1 for i in get_bond_feature_dims()]
                    topodis = matrix_topodis[i][j]
                    edge_feature.append(topodis if topodis < 9 else 9)
                    edge_feature_float = [matrix_3ddis[i][j]]

                    edges_list.append((i, j))
                    edge_features_list.append(edge_feature)
                    edge_features_float_list.append(edge_feature_float)
        if edges_list:
            edge_index = np.hstack((graph["edge_index"], np.array(edges_list, dtype=np.int64).T))
            edge_attr = np.vstack((graph["edge_feat"], np.array(edge_features_list)))
            edge_attr_float = np.vstack((edge_attr_float, np.array(edge_features_float_list)))
            graph["edge_index"] = edge_index
            graph["edge_feat"] = edge_attr

    graph["edge_feat_float"] = edge_attr_float
    graph['num_edges'] = graph["edge_index"].shape[1]

    if gen_aux_task:
        bond_angle_index = get_bond_angle_index(graph['edge_index'])
        bond_angle = get_bond_angle(x_position, bond_angle_index)
        graph['bond_angle_index'] = bond_angle_index
        graph['bond_angle'] = bond_angle

    return graph

# node feature
#TODO:Acceptor, Donor ; covalent radius ; Atom radius
# return (N,6) list
def get_ring_size(mol):
    # smiles = 'CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O'
    # test smiles that containing atoms included in two rings
    # mol = Chem.MolFromSmiles(smiles)
    # print(mol.get)
    rings = mol.GetRingInfo()
    rings_info = []
    for r in rings.AtomRings():
        rings_info.append(r)
        # print(f"r:{r}")
    # print("rings",rings)
    ring_list = []
    for atom in mol.GetAtoms():
        # atom_ring_list = []
        atom_result = []
        # num_atom_rings = rings.NumAtomRings(atom.GetIdx())
        # print(f"atom index: {atom.GetIdx()}")
        for ringsize in range(3, 9):
            # atom_ring_list.append(rings.IsAtomInRingOfSize(atom.GetIdx(), ringsize))
            num_of_ring_at_ringsize = 0
            for r in rings_info:
                if len(r) == ringsize and atom.GetIdx() in r:
                    num_of_ring_at_ringsize +=1
            if num_of_ring_at_ringsize > 8:
                num_of_ring_at_ringsize = 9
            atom_result.append(num_of_ring_at_ringsize)
        
        # print(atom_result)
        ring_list.append(atom_result)
        
    return ring_list

def get_partial_charge(mol):
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    atoms = mol.GetAtoms()
    charge_list = []
    for atom in atoms:
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float('inf'):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10
        charge_list.append(pc)
    return charge_list

def get_h_num(mol):
    atoms = mol.GetAtoms()
    h_list = []
    for atom in atoms:
        h_list.append(atom.GetTotalNumHs())
    return h_list

def get_van_der_waals_radis(mol):
    peroid_table = Chem.GetPeriodicTable()
    radis_list = []
    atoms = mol.GetAtoms()
    for atom in atoms:
        radis_list.append(peroid_table.GetRvdw(atom.GetAtomicNum()))
    return radis_list

def get_valence_of_out_shell(mol):
    peroid_table = Chem.GetPeriodicTable()
    valence_list = []
    atoms = mol.GetAtoms()
    for atom in atoms:
        valence_out_shell = peroid_table.GetNOuterElecs(atom.GetAtomicNum())
        if valence_out_shell > 8:
            valence_out_shell = 9
        valence_list.append(valence_out_shell)
    return valence_list

# edge feature 
# TODO: shortest path bonds; top_path_length: dimension handling ; 
def edge_same_ring(mol):
    rings = mol.GetRingInfo()
    bonds = mol.GetBonds()
    # print(f"bonds are {bonds}")
    same_ring_list = []
    for bond in bonds:
        same_ring = False
        bond_idx = bond.GetIdx()
        # print(f"bond_idx are {bond_idx}")
        for r in rings.BondRings():
            if bond_idx in r:
                same_ring = True
                break
        same_ring_list.append(same_ring)
    return same_ring_list

def edge_top_length(mol):
    top_length = Chem.rdmolops.GetDistanceMatrix(mol)
    return top_length

def edge_geo_distance(mol):
    m3d = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(m3d, randomSeed=1)
    _3dm = Chem.rdmolops.Get3DDistanceMatrix(m3d)
    return _3dm

# def edge_expanded_distane(mol):
    
def get_word_dict(cur_list, cur_dict):
    for element in cur_list:
        # print(f"ele:{element}")
        if element not in cur_dict:
            cur_dict[element] = 1
    return cur_dict

def get_bond_angle_index(edge_index):
    """
    edge_index: (2, E)
    bond_angle_index: (3, *)
    """
    def _add_item(
            node_i_indices, node_j_indices, node_k_indices,
            node_i_index, node_j_index, node_k_index):
        node_i_indices += [node_i_index, node_k_index]
        node_j_indices += [node_j_index, node_j_index]
        node_k_indices += [node_k_index, node_i_index]

    E = edge_index.shape[1]
    node_i_indices = []
    node_j_indices = []
    node_k_indices = []
    for edge_i in range(E - 1):
        for edge_j in range(edge_i + 1, E):
            a0, a1 = edge_index[:, edge_i]
            b0, b1 = edge_index[:, edge_j]
            if a0 == b0 and a1 == b1:
                continue
            if a0 == b1 and a1 == b0:
                continue
            if a0 == b0:
                _add_item(node_i_indices, node_j_indices, node_k_indices,
                          a1, a0, b1)
            if a0 == b1:
                _add_item(node_i_indices, node_j_indices, node_k_indices,
                          a1, a0, b0)
            if a1 == b0:
                _add_item(node_i_indices, node_j_indices, node_k_indices,
                          a0, a1, b1)
            if a1 == b1:
                _add_item(node_i_indices, node_j_indices, node_k_indices,
                          a0, a1, b0)
    node_ijk = np.array([node_i_indices, node_j_indices, node_k_indices])
    uniq_node_ijk = np.unique(node_ijk, axis=1).astype('int64')     # (3, *)
    return uniq_node_ijk

def get_bond_angle(atom_positions, bond_angle_index):
    """
    atom_positions: (N, 3)
    bond_angle_index: (3, A)
    bond_angle: (A)
    """
    def _get_angle(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
        vec2 = vec2 / (norm2 + 1e-5)
        angle = np.arccos(np.dot(vec1, vec2))
        return angle
    node_i, node_j, node_k = bond_angle_index
    node_i_pos = atom_positions[node_i]
    node_j_pos = atom_positions[node_j]
    node_k_pos = atom_positions[node_k]

    v1 = node_i_pos - node_j_pos
    v2 = node_k_pos - node_j_pos
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-5
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-5
    v1 /= norm1
    v2 /= norm2
    bond_angle = np.arccos(np.sum(v1 * v2, 1))
    return np.nan_to_num(bond_angle)



if __name__ == '__main__':
    graph = smiles2graph('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
    print(graph)
