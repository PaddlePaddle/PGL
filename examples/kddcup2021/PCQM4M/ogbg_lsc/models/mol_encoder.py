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
import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist

from pgl.utils.logger import log

import features.local_feature as MolFeat

full_atom_feature_dims = MolFeat.get_atom_feature_dims()
full_bond_feature_dims = MolFeat.get_bond_feature_dims()

def batch_norm_1d(num_channels):
    if dist.get_world_size() > 1:
        return nn.SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm1D(num_channels))
    else:
        return nn.BatchNorm1D(num_channels)

class AtomEncoder(nn.Layer):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        log.info("atom encoder type is %s" % self.__class__.__name__)

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
        log.info("bond encoder type is %s" % self.__class__.__name__)

        self.bond_embedding_list = nn.LayerList()

        for i, dim in enumerate(full_bond_feature_dims):
            weight_attr = nn.initializer.XavierUniform()
            emb = paddle.nn.Embedding(dim, emb_dim, weight_attr=weight_attr)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(len(full_bond_feature_dims)):
            feat = edge_attr[:, i]
            bond_embedding += self.bond_embedding_list[i](feat)

        return bond_embedding

class CatAtomEncoder(nn.Layer):
    def __init__(self, emb_dim):
        super(CatAtomEncoder, self).__init__()
        log.info("atom encoder type is %s" % self.__class__.__name__)

        self.atom_embedding_list = nn.LayerList()

        for i, dim in enumerate(full_atom_feature_dims):
            weight_attr = nn.initializer.XavierUniform()
            emb = paddle.nn.Embedding(dim, emb_dim, weight_attr=weight_attr)
            self.atom_embedding_list.append(emb)

        self.mlp = nn.Sequential(nn.Linear(len(full_atom_feature_dims) * emb_dim, 2 * emb_dim),
                        batch_norm_1d(2 * emb_dim),
                        nn.Swish(),
                        nn.Linear(2 * emb_dim, emb_dim))
        #  channels = [len(full_atom_feature_dims) * emb_dim, 2 * emb_dim, emb_dim]
        #  self.mlp = MLP(channels, norm="batch", last_lin=True)

    def forward(self, x):
        x_embedding = []
        for i in range(x.shape[1]):
            x_embedding.append(self.atom_embedding_list[i](x[:, i]))

        x_embedding = paddle.concat(x_embedding, axis=1)

        return self.mlp(x_embedding)


class CatBondEncoder(nn.Layer):
    def __init__(self, emb_dim):
        super(CatBondEncoder, self).__init__()
        log.info("bond encoder type is %s" % self.__class__.__name__)

        self.bond_embedding_list = nn.LayerList()

        for i, dim in enumerate(full_bond_feature_dims):
            weight_attr = nn.initializer.XavierUniform()
            emb = paddle.nn.Embedding(dim, emb_dim, weight_attr=weight_attr)
            self.bond_embedding_list.append(emb)

        self.mlp = nn.Sequential(nn.Linear(len(full_bond_feature_dims) * emb_dim, 2 * emb_dim),
                        batch_norm_1d(2 * emb_dim),
                        nn.Swish(),
                        nn.Linear(2 * emb_dim, emb_dim))

    def forward(self, edge_attr):
        bond_embedding = []
        for i in range(edge_attr.shape[1]):
            bond_embedding.append(self.bond_embedding_list[i](edge_attr[:, i]))

        bond_embedding = paddle.concat(bond_embedding, axis=1)

        return self.mlp(bond_embedding)

class AtomEncoderFloat(nn.Layer):
    def __init__(self, emb_dim):
        super(AtomEncoderFloat, self).__init__()

        num_center = 21
        centers1 = np.linspace(-1, 4, num_center).reshape(1, -1)

        self.centers1 = self.create_parameter(shape=centers1.shape,
                             dtype="float32",
                             default_initializer=nn.initializer.Assign(centers1))
        self.centers1.stop_gradient=True

        centers2 = np.linspace(1, 3, num_center).reshape(1, -1)
        self.centers2 = self.create_parameter(shape=centers2.shape,
                             dtype="float32",
                             default_initializer=nn.initializer.Assign(centers2))
        self.centers2.stop_gradient=True

        self.width = 0.5

        num_float_feat = 2
        self.atom_embedding_list = nn.LayerList()
        for _i in range(num_float_feat):
            emb = nn.Linear(num_center, emb_dim, bias_attr=False)
            self.atom_embedding_list.append(emb)

    def forward(self, x_float):
        x_embedding = 0
        for i in range(x_float.shape[1]):
            x = x_float[:, i]
            x =  paddle.reshape(x, [-1, 1])
            if i == 0:
                gaussian_expansion = paddle.exp(-(x - self.centers1)**2 / self.width**2)
            elif i == 1:
                gaussian_expansion = paddle.exp(-(x - self.centers2)**2 / self.width**2)

            x_embedding += self.atom_embedding_list[i](gaussian_expansion)
        return x_embedding




if __name__ == '__main__':
    from ogb.graphproppred import GraphPropPredDataset
    dataset = GraphPropPredDataset(name='ogbg-molpcba')
    atom_enc = AtomEncoder(100)
    bond_enc = BondEncoder(100)

    print(atom_enc(dataset[0].x))
    print(bond_enc(dataset[0].edge_attr))
