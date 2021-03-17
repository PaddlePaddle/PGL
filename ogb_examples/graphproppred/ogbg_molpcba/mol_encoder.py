# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import paddle
import paddle.nn as nn
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


class AtomEncoder(nn.Layer):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = nn.LayerList()

        for i, dim in enumerate(full_atom_feature_dims):
            weight_attr = nn.initializer.XavierUniform()
            emb = paddle.nn.Embedding(dim, emb_dim, weight_attr=weight_attr)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(nn.Layer):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = nn.LayerList()

        for i, dim in enumerate(full_bond_feature_dims):
            weight_attr = nn.initializer.XavierUniform()
            emb = paddle.nn.Embedding(dim, emb_dim, weight_attr=weight_attr)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


if __name__ == '__main__':
    from ogb.graphproppred import GraphPropPredDataset
    dataset = GraphPropPredDataset(name='ogbg-molpcba')
    atom_enc = AtomEncoder(100)
    bond_enc = BondEncoder(100)

    print(atom_enc(dataset[0].x))
    print(bond_enc(dataset[0].edge_attr))
