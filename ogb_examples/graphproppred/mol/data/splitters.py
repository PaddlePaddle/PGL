# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import logging
from random import random
import pandas as pd
import numpy as np
from itertools import compress

import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from rdkit.Chem.Scaffolds import MurckoScaffold

import pgl
from pgl.utils import paddle_helper
try:
    from dataset.Dataset import Subset
    from dataset.Dataset import ChemDataset
except:
    from Dataset import Subset
    from Dataset import ChemDataset

log = logging.getLogger("logger")


def random_split(dataset, args):
    total_precent = args.frac_train + args.frac_valid + args.frac_test
    np.testing.assert_almost_equal(total_precent, 1.0)

    length = len(dataset)
    perm = list(range(length))
    np.random.shuffle(perm)
    num_train = int(args.frac_train * length)
    num_valid = int(args.frac_valid * length)
    num_test = int(args.frac_test * length)

    train_indices = perm[0:num_train]
    valid_indices = perm[num_train:(num_train + num_valid)]
    test_indices = perm[(num_train + num_valid):]
    assert (len(train_indices) + len(valid_indices) + len(test_indices)
            ) == length

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, valid_dataset, test_dataset


def scaffold_split(dataset, args, return_smiles=False):
    total_precent = args.frac_train + args.frac_valid + args.frac_test
    np.testing.assert_almost_equal(total_precent, 1.0)

    smiles_list_file = os.path.join(args.data_dir, "smiles.csv")
    smiles_list = pd.read_csv(smiles_list_file, header=None)[0].tolist()

    non_null = np.ones(len(dataset)) == 1
    smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=True)
        #  scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {
        key: sorted(value)
        for key, value in all_scaffolds.items()
    }
    all_scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(),
            key=lambda x: (len(x[1]), x[1][0]),
            reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = args.frac_train * len(smiles_list)
    valid_cutoff = (args.frac_train + args.frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(
                    scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0
    #  log.info(len(scaffold_set))
    #  log.info(["train_idx", train_idx])
    #  log.info(["valid_idx", valid_idx])
    #  log.info(["test_idx", test_idx])

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    if return_smiles:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]

        return train_dataset, valid_dataset, test_dataset, (
            train_smiles, valid_smiles, test_smiles)

    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.realpath(__file__))
    proj_path = os.path.join(file_path, '../')
    sys.path.append(proj_path)
    from utils.config import Config
    from dataset.Dataset import Subset
    from dataset.Dataset import ChemDataset

    config_file = "./finetune_config.yaml"
    args = Config(config_file)
    log.info("loading dataset")
    dataset = ChemDataset(args)

    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, args)

    log.info("Train Examples: %s" % len(train_dataset))
    log.info("Val Examples: %s" % len(valid_dataset))
    log.info("Test Examples: %s" % len(test_dataset))
    import ipdb
    ipdb.set_trace()
    log.info("preprocess finish")
