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
"""MolEncoder for ogb
"""
import paddle.fluid as fluid
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


class AtomEncoder(object):
    """AtomEncoder for encoding node features"""

    def __init__(self, name, emb_dim):
        self.emb_dim = emb_dim
        self.name = name

    def __call__(self, x):
        atom_feature = get_atom_feature_dims()
        atom_input = fluid.layers.split(
            x, num_or_sections=len(atom_feature), dim=-1)
        outputs = None
        count = 0
        for _x, _atom_input_dim in zip(atom_input, atom_feature):
            count += 1
            emb = fluid.layers.embedding(
                _x,
                size=(_atom_input_dim, self.emb_dim),
                param_attr=fluid.ParamAttr(
                    name=self.name + '_atom_feat_%s' % count))
            if outputs is None:
                outputs = emb
            else:
                outputs = outputs + emb
        return outputs


class BondEncoder(object):
    """Bond for encoding edge features"""

    def __init__(self, name, emb_dim):
        self.emb_dim = emb_dim
        self.name = name

    def __call__(self, x):
        bond_feature = get_bond_feature_dims()
        bond_input = fluid.layers.split(
            x, num_or_sections=len(bond_feature), dim=-1)
        outputs = None
        count = 0
        for _x, _bond_input_dim in zip(bond_input, bond_feature):
            count += 1
            emb = fluid.layers.embedding(
                _x,
                size=(_bond_input_dim, self.emb_dim),
                param_attr=fluid.ParamAttr(
                    name=self.name + '_bond_feat_%s' % count))
            if outputs is None:
                outputs = emb
            else:
                outputs = outputs + emb
        return outputs
