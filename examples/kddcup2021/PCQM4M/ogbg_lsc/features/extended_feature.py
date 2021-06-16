# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from rdkit import Chem
import rdkit.Chem.rdPartialCharges

from ogb.utils.features import (allowable_features, atom_to_feature_vector,
                                atom_feature_vector_to_dict,
                                bond_feature_vector_to_dict)

from local_feature import bond_to_feature_vector, get_bond_feature_dims


def shortest_path(mol):
    matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    return matrix


def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

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

    x = np.array(atom_features_list, dtype=np.int64)
    x_float = np.array(atom_features_float_list, dtype=np.float)

    # bonds
    num_bond_features = 4  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) == 0:  # mol has bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
    else:
        # real edges
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)
            edge_feature.append(1)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

        # # k-hop edges
        # matrix = shortest_path(mol)
        # edges_list = []
        # edge_features_list = []
        # for i in range(len(matrix)):
        #     for j in range(len(matrix[0])):
        #         if matrix[i][j] == 2:
        #             # virtual feature for k-hop connected bond, value = len()-1
        #             edge_feature = [i - 1 for i in get_bond_feature_dims()]
        #             edge_feature.append(2)

        #             edges_list.append((i, j))
        #             edge_features_list.append(edge_feature)
        # if edges_list:
        #     edge_index = np.hstack((edge_index, np.array(edges_list, dtype = np.int64).T))
        #     edge_attr = np.vstack((edge_attr, np.array(edge_features_list)))

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['node_feat_float'] = x_float
    graph['num_nodes'] = len(x)
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
                    num_of_ring_at_ringsize += 1
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


if __name__ == '__main__':
    graph = smiles2graph(
        'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
    print(graph)
