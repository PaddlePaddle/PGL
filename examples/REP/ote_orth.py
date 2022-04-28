# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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


class OrthOTE():
    def __init__(self, relation_feat, ote_size):
        self.use_scale = True
        self.num_elem = ote_size
        self.scale_type = 2
        if isinstance(relation_feat, str):
            relation_emb = np.load(relation_feat)
        else:
            relation_emb = relation_feat
        relation_emb = paddle.to_tensor(relation_emb)
        self.orth_relation_emb = self.orth_rel_embedding(relation_emb)
        self.orth_relation_emb_mat = self.orth_reverse_mat(
            self.orth_relation_emb)

    def orth_reverse_mat(self, rel_embeddings):
        rel_shape = rel_embeddings.shape
        if self.use_scale:
            rel_emb = rel_embeddings.reshape(
                [-1, self.num_elem, self.num_elem + 1])
            rel_mat = rel_emb[:, :, :self.num_elem].transpose([0, 2, 1])
            rel_scale = self.reverse_scale(rel_emb[:, :, self.num_elem:])
            rel_embeddings = paddle.concat(
                [rel_mat, rel_scale], axis=-1).reshape(rel_shape)
        else:
            rel_embeddings = rel_embeddings.reshape(
                [-1, self.num_elem, self.num_elem])
            rel_embeddings = rel_embeddings.transpose(
                [0, 2, 1]).reshape(rel_shape)
        return rel_embeddings

    def reverse_scale(self, scale, eps=1e-9):
        if self.scale_type == 1:
            return 1 / (scale.abs() + eps)
        if self.scale_type == 2:
            return -scale
        raise ValueError("Scale Type %d is not supported!" % self.scale_type)

    def orth_embedding(self, embeddings, eps=1e-18):
        assert embeddings.shape[1] == self.num_elem
        assert embeddings.shape[2] == (self.num_elem + int(self.use_scale))
        if self.use_scale:
            emb_scale = embeddings[:, :, -1]
            embeddings = embeddings[:, :, :self.num_elem]

        u = [embeddings[:, 0]]
        uu = [0] * self.num_elem
        uu[0] = (u[0] * u[0]).sum(axis=-1)
        u_d = embeddings[:, 1:]
        for i in range(1, self.num_elem):
            tmp = (embeddings[:, i:] * u[i - 1].unsqueeze(axis=1)).sum(axis=-1)
            tmp = (tmp / uu[i - 1].unsqueeze(axis=1)).unsqueeze(axis=-1)
            u_d = u_d - u[-1].unsqueeze(axis=1) * tmp
            u_i = u_d[:, 0]
            if u_d.shape[1] > 1:
                u_d = u_d[:, 1:]
            uu[i] = (u_i * u_i).sum(axis=-1)
            u.append(u_i)
        u = paddle.stack(u, axis=1)
        u = u / u.norm(axis=-1, keepdim=True, p=2)
        if self.use_scale:
            u = paddle.concat([u, emb_scale.unsqueeze(-1)], axis=-1)
        return u

    def orth_rel_embedding(self, relation_embedding):
        rel_emb_shape = relation_embedding.shape
        relation_embedding = relation_embedding.reshape(
            [-1, self.num_elem, self.num_elem + int(self.use_scale)])
        relation_embedding = self.orth_embedding(relation_embedding)
        relation_embedding = relation_embedding.reshape(rel_emb_shape)
        return relation_embedding
