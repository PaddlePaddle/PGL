import os
from rdkit import Chem

class CD:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def read_sdf_withH(path):
    mol_supplier = Chem.SDMolSupplier()
    input_sdf = open(path, 'r')
    input_string = ''.join([line for line in input_sdf])
    mol_supplier.SetData(input_string, sanitize=False, removeHs=False)
    return mol_supplier

def get_mol_withH_from_sdf_string(sdf_string):
    mol_supplier = Chem.SDMolSupplier()
    mol_supplier.SetData(sdf_string, sanitize=False, removeHs=False)
#     mol_supplier.SetData(sdf_string, sanitize=False, removeHs=True)
#     print("Set data has removed the HS")
    return mol_supplier[0]

def read_csv_raw(path, max_line=None):
    with open(path, 'r') as f:
        f.readline()    # drop the first line
        line_num = 0
        rtn_list = []
        for line in f:
            i, smiles, value = line.split(',')
            if value == "\n":
                value = 0
            rtn_list.append((int(i), smiles, float(value)))

            line_num += 1
            if max_line and line_num == max_line:
                break
        return rtn_list

def pos2pyscf_input(mol):
    conf = mol.GetConformer()
    mol_string = []
    for atom in mol.GetAtoms():
        mol_string += [atom.GetSymbol() + " " + str(conf.GetAtomPosition(atom.GetIdx()).x)\
            + " " + str(conf.GetAtomPosition(atom.GetIdx()).y)\
            + " " + str(conf.GetAtomPosition(atom.GetIdx()).z)]
    mol_string = ';'.join(mol_string)
    return mol_string

def update_pos(mol, dft_matrix):
    conf = mol.GetConformer()
    for i, atom in enumerate(dft_matrix):
        conf.SetAtomPosition(i, dft_matrix[i])

def flatten_sdf_string(sdf_string):  
    return sdf_string.replace('\n', ';;;')

def deflatten_sdf_string(sdf_string):
    return sdf_string.replace(';;;', '\n')
