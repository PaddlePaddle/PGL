# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import math

import numpy as np
import paddle
import paddle.nn as nn
from numpy.random import RandomState
from paddle.nn.functional import log_sigmoid

from models.numpy_embedding import NumPyEmbedding


class ScoreFunc(object):
    """Abstract implementation of score function
    """

    def __init__(self):
        super(ScoreFunc, self).__init__()
        self.embed_epsilon = 2.0

    def __call__(self, head, rel, tail):
        raise NotImplementedError(
            'score function from head, relation to tail not implemented')

    def get_neg_score(self, head, rel, tail, neg_head=False):
        """Score function from tail, relation to head
        """
        raise NotImplementedError(
            'score function from tail, relation to head not implemented')

    def get_init_weight(self):
        """Initialization for embeddings
        Return initialization range (float) or weights (np.ndarray) for embeddings
        """
        raise NotImplementedError('weight initialization not implemented')


class TransEScore(ScoreFunc):
    """
    TransE: Translating embeddings for modeling multi-relational data.
    https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf
    """

    def __init__(self, gamma):
        super(TransEScore, self).__init__()
        self.gamma = gamma

    def __call__(self, head, rel, tail):
        head = head + rel
        score = self.gamma - paddle.norm(head - tail, p=2, axis=-1)
        return score

    def get_init_weight(self, embed_dim):
        return (self.gamma + self.embed_epsilon) / embed_dim

    def get_neg_score(self, head, rel, tail, neg_head=False):
        if neg_head:
            tail = tail - rel
            score = self.gamma - self.cdist(tail, head)
        else:
            head = head + rel
            score = self.gamma - self.cdist(head, tail)
        return score

    def cdist(self, a, b):
        """Euclidean distance
        """
        a_s = paddle.norm(a, p=2, axis=-1).pow(2)
        b_s = paddle.norm(b, p=2, axis=-1).pow(2)
        dist_score = -2 * paddle.bmm(a, b.transpose([0, 2, 1]))
        dist_score = dist_score + b_s.unsqueeze(-2) + a_s.unsqueeze(-1)
        dist_score = paddle.sqrt(paddle.clip(dist_score, min=1e-30))
        return dist_score


class DistMultScore(ScoreFunc):
    """DistMult
    https://arxiv.org/abs/1412.6575
    """

    def __init__(self):
        super(DistMultScore, self).__init__()

    def __call__(self, head, rel, tail):
        score = head * rel * tail
        score = paddle.sum(score, axis=-1)
        return score

    def get_neg_score(self, head, rel, tail, neg_head=False):
        num_chunks = head.shape[0]
        if neg_head:
            tail = tail * rel
            score = paddle.bmm(tail, head.transpose([0, 2, 1]))
            return score
        else:
            head = head * rel
            score = paddle.bmm(head, tail.transpose([0, 2, 1]))
            return score


class RotatEScore(ScoreFunc):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.
    https://arxiv.org/abs/1902.10197
    """

    def __init__(self, gamma, embed_dim):
        super(RotatEScore, self).__init__()
        self.epsilon = 1e-12
        self.gamma = gamma
        self.emb_init = self.get_init_weight(embed_dim)

    def __call__(self, head, rel, tail):
        re_head, im_head = paddle.chunk(head, chunks=2, axis=-1)
        re_tail, im_tail = paddle.chunk(tail, chunks=2, axis=-1)
        phase_rel = rel / (self.emb_init / np.pi)
        re_rel, im_rel = paddle.cos(phase_rel), paddle.sin(phase_rel)

        re_score = re_rel * re_head - im_rel * im_head
        im_score = re_rel * im_head + im_rel * re_head
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = paddle.sqrt(re_score * re_score + im_score * im_score +
                            self.epsilon)
        score = self.gamma - paddle.sum(score, axis=-1)
        return score

    def get_init_weight(self, embed_dim):
        return (self.gamma + self.embed_epsilon) / embed_dim

    def get_neg_score(self, head, rel, tail, neg_head=False):
        num_chunks = head.shape[0]

        if neg_head:
            chunk_size = tail.shape[1]
            neg_sample_size = head.shape[1]

            re_tail, im_tail = paddle.chunk(tail, chunks=2, axis=-1)
            phase_rel = rel / (self.emb_init / np.pi)
            re_rel, im_rel = paddle.cos(phase_rel), paddle.sin(phase_rel)

            re_score = re_rel * re_tail + im_rel * im_tail
            im_score = re_rel * im_tail - im_rel * re_tail
            score = paddle.concat([re_score, im_score], axis=-1)
            score = paddle.reshape(score, [num_chunks, chunk_size, 1, -1])
            head = paddle.reshape(head, [num_chunks, 1, neg_sample_size, -1])

            score = paddle.tile(score, repeat_times=[1, 1, neg_sample_size, 1])
            head = paddle.tile(head, repeat_times=[1, chunk_size, 1, 1])

            score = score - head
            re_score, im_score = paddle.chunk(score, chunks=2, axis=-1)

            score = paddle.sqrt(re_score * re_score + im_score * im_score +
                                self.epsilon)
            score = self.gamma - paddle.sum(score, axis=-1)
        else:
            chunk_size = head.shape[1]
            neg_sample_size = tail.shape[1]

            re_head, im_head = paddle.chunk(head, chunks=2, axis=-1)
            phase_rel = rel / (self.emb_init / np.pi)
            re_rel, im_rel = paddle.cos(phase_rel), paddle.sin(phase_rel)

            re_score = re_rel * re_head - im_rel * im_head
            im_score = re_rel * im_head + im_rel * re_head
            score = paddle.concat([re_score, im_score], axis=-1)
            score = paddle.reshape(score, [num_chunks, chunk_size, 1, -1])
            tail = paddle.reshape(tail, [num_chunks, 1, neg_sample_size, -1])

            score = paddle.tile(score, repeat_times=[1, 1, neg_sample_size, 1])
            tail = paddle.tile(tail, repeat_times=[1, chunk_size, 1, 1])

            score = score - tail
            re_score, im_score = paddle.chunk(score, chunks=2, axis=-1)

            score = paddle.sqrt(re_score * re_score + im_score * im_score +
                                self.epsilon)
            score = self.gamma - paddle.sum(score, axis=-1)

        return score


class ComplExScore(ScoreFunc):
    """ComplEx
    https://arxiv.org/abs/1606.0635
    """

    def __init__(self):
        super(ComplExScore, self).__init__()

    def __call__(self, head, rel, tail):
        re_head, im_head = paddle.chunk(head, chunks=2, axis=-1)
        re_tail, im_tail = paddle.chunk(tail, chunks=2, axis=-1)
        re_rel, im_rel = paddle.chunk(rel, chunks=2, axis=-1)

        score = re_head * re_tail * re_rel + im_head * im_tail * re_rel \
            + re_head * im_tail * im_rel - im_head * re_tail * im_rel
        score = paddle.sum(score, axis=-1)
        return score

    def get_neg_score(self, head, rel, tail, neg_head=False):
        if neg_head:
            re_tail, im_tail = paddle.chunk(tail, chunks=2, axis=-1)
            re_rel, im_rel = paddle.chunk(rel, chunks=2, axis=-1)
            re_emb = re_tail * re_rel + im_tail * im_rel
            im_emb = im_tail * re_rel - re_tail * im_rel
            complex_emb = paddle.concat([re_emb, im_emb], axis=-1)
            complex_emb = paddle.reshape(complex_emb, tail.shape)
            score = paddle.bmm(complex_emb, head.transpose([0, 2, 1]))
            return score
        else:
            re_head, im_head = paddle.chunk(head, chunks=2, axis=-1)
            re_rel, im_rel = paddle.chunk(rel, chunks=2, axis=-1)
            re_emb = re_head * re_rel - im_head * im_rel
            im_emb = re_head * im_rel + im_head * re_rel
            complex_emb = paddle.concat([re_emb, im_emb], axis=-1)
            complex_emb = paddle.reshape(complex_emb, head.shape)
            score = paddle.bmm(complex_emb, tail.transpose([0, 2, 1]))
            return score


class QuatEScore(ScoreFunc):
    """QuatE score
    https://arxiv.org/abs/1904.10281
    """

    def __init__(self):
        super(QuatEScore, self).__init__()

    def __call__(self, head, rel, tail):
        A, B, C, D = self._get_part_score(head, rel)
        tails = paddle.chunk(tail, chunks=4, axis=-1)

        score = (A * tails[0] + B * tails[1] + C * tails[2] + D * tails[3])
        score = paddle.sum(score, axis=-1)
        return score

    def get_neg_score(self, head, rel, tail, neg_head=False):
        if neg_head:
            return self._get_neg_score(tail, rel, head)
        else:
            return self._get_neg_score(head, rel, tail)

    def _get_part_score(self, a, b):
        a = paddle.chunk(a, chunks=4, axis=-1)
        b = paddle.chunk(b, chunks=4, axis=-1)

        denominator_b = paddle.sqrt(b[0]**2 + b[1]**2 + b[2]**2 + b[3]**2)
        for i in range(4):
            b[i] = b[i] / denominator_b

        A = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        B = a[0] * b[1] + b[0] * a[1] + a[2] * b[3] - b[2] * a[3]
        C = a[0] * b[2] + b[0] * a[2] + a[3] * b[1] - b[3] * a[1]
        D = a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - b[1] * a[2]

        return A, B, C, D

    def _get_neg_score(self, pos_e, pos_r, neg_e):
        num_chunks = pos_e.shape[0]
        chunk_size = pos_e.shape[1]
        neg_sample_size = neg_e.shape[1]

        A, B, C, D = self._get_part_score(pos_e, pos_r)

        A = paddle.reshape(A, (num_chunks, chunk_size, -1))
        B = paddle.reshape(B, (num_chunks, chunk_size, -1))
        C = paddle.reshape(C, (num_chunks, chunk_size, -1))
        D = paddle.reshape(D, (num_chunks, chunk_size, -1))

        neg_e = paddle.reshape(neg_e, (num_chunks, neg_sample_size, -1))
        neg_e = neg_e.transpose((0, 2, 1))
        neg_e = paddle.chunk(neg_e, chunks=4, axis=1)

        score = paddle.bmm(A, neg_e[0]) + paddle.bmm(B, neg_e[1]) + \
                paddle.bmm(C, neg_e[2]) + paddle.bmm(D, neg_e[3])
        return score

    def get_init_weight(self, num_embed, embed_dim):
        init_value = 1. / np.sqrt(2 * num_embed)
        rng = RandomState(123)

        kernel_shape = (num_embed, embed_dim)
        num_weights = np.prod(kernel_shape)
        v_i = np.random.uniform(0., 1., num_weights)
        v_j = np.random.uniform(0., 1., num_weights)
        v_k = np.random.uniform(0., 1., num_weights)

        for i in range(num_weights):
            norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2) + 1e-4
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)

        modulus = rng.uniform(-init_value, init_value, size=kernel_shape)
        phase = rng.uniform(-np.pi, np.pi, size=kernel_shape)

        w_r = modulus * np.cos(phase)
        w_i = modulus * v_i * np.sin(phase)
        w_j = modulus * v_j * np.sin(phase)
        w_k = modulus * v_k * np.sin(phase)

        return np.concatenate([w_r, w_i, w_j, w_k], axis=-1).astype(np.float32)

    def get_regularization(self, head, rel, tail):
        heads = paddle.chunk(head, 4, axis=-1)
        tails = paddle.chunk(tail, 4, axis=-1)
        rels = paddle.chunk(rel, 4, axis=-1)

        reg_ents = paddle.mean(paddle.abs(heads[0]) ** 2) + \
            paddle.mean(paddle.abs(heads[1]) ** 2) + \
            paddle.mean(paddle.abs(heads[2]) ** 2) + \
            paddle.mean(paddle.abs(heads[3]) ** 2) + \
            paddle.mean(paddle.abs(tails[0]) ** 2) + \
            paddle.mean(paddle.abs(tails[1]) ** 2) + \
            paddle.mean(paddle.abs(tails[2]) ** 2) + \
            paddle.mean(paddle.abs(tails[3]) ** 2)

        reg_rels = paddle.mean(paddle.abs(rels[0]) ** 2) + \
            paddle.mean(paddle.abs(rels[1]) ** 2) + \
            paddle.mean(paddle.abs(rels[2]) ** 2) + \
            paddle.mean(paddle.abs(rels[3]) ** 2)

        return reg_ents, reg_rels


class OTEScore(ScoreFunc):
    """OTE score
    """

    def __init__(self, gamma, num_elem, scale_type=0):
        super(OTEScore, self).__init__()
        self.gamma = gamma
        self.num_elem = num_elem
        self.scale_type = scale_type
        self.use_scale = self.scale_type > 0

    def __call__(self, head, rel, tail):
        rel = self.orth_rel_embedding(rel)
        score = self.score(head, rel, tail)
        return self.gamma - score

    def get_init_weight(self, embed_dim):
        if self.use_scale:
            value = 1. / math.sqrt(embed_dim)
        else:
            value = (self.gamma + self.embed_epsilon) / embed_dim

        return value

    def inverse(self, head, rel, tail):
        rel = self.orth_rel_embedding(rel)
        rel = self.orth_reverse_mat(rel)
        score = self.score(tail, rel, head)
        return self.gamma - score

    def score(self, inputs, inputs_rel, inputs_last):
        inputs_size = inputs.shape
        assert inputs_size[:-1] == inputs_rel.shape[:-1]
        num_dim = inputs_size[-1]
        inputs = inputs.reshape([-1, 1, self.num_elem])
        if self.use_scale:
            rel = inputs_rel.reshape([-1, self.num_elem, self.num_elem + 1])
            scale = self.get_scale(rel[:, :, self.num_elem:])
            scale = scale / scale.norm(axis=-1, p=2, keepdim=True)
            rel_scale = rel[:, :, :self.num_elem] * scale
            outputs = paddle.bmm(inputs, rel_scale)
        else:
            rel = inputs_rel.reshape([-1, self.num_elem, self.num_elem])
            outputs = paddle.bmm(inputs, rel)
        outputs = outputs.reshape(inputs_size)
        outputs = outputs - inputs_last
        outputs_size = outputs.shape
        num_dim = outputs_size[-1]
        outputs = outputs.reshape([-1, self.num_elem])
        scores = outputs.norm(
            p=2, axis=-1).reshape([-1, num_dim // self.num_elem]).sum(
                axis=-1).reshape(outputs_size[:-1])
        return scores

    def neg_score(self, inputs, inputs_rel, inputs_last, neg_sample_size,
                  chunk_size):
        inputs_size = inputs.shape
        assert inputs_size[:-1] == inputs_rel.shape[:-1]
        num_dim = inputs_size[-1]
        inputs = inputs.reshape([-1, 1, self.num_elem])
        if self.use_scale:
            rel = inputs_rel.reshape([-1, self.num_elem, self.num_elem + 1])
            scale = self.get_scale(rel[:, :, self.num_elem:])
            scale = scale / scale.norm(axis=-1, p=2, keepdim=True)
            rel_scale = rel[:, :, :self.num_elem] * scale
            outputs = paddle.bmm(inputs, rel_scale)
        else:
            rel = inputs_rel.reshape([-1, self.num_elem, self.num_elem])
            outputs = paddle.bmm(inputs, rel)
        outputs = outputs.reshape([-1, chunk_size, 1, inputs_size[-1]])
        inputs_last = inputs_last.reshape(
            [-1, 1, neg_sample_size, inputs_size[-1]])
        outputs = outputs - inputs_last
        outputs_size = outputs.shape
        num_dim = outputs_size[-1]
        outputs = outputs.reshape([-1, self.num_elem])
        scores = outputs.norm(
            p=2, axis=-1).reshape([-1, num_dim // self.num_elem]).sum(
                axis=-1).reshape(outputs_size[:-1])
        return scores

    def get_scale(self, scale):
        if self.scale_type == 1:
            return scale.abs()
        if self.scale_type == 2:
            return scale.exp()
        raise ValueError("Scale Type %d is not supported!" % self.scale_type)

    def reverse_scale(self, scale, eps=1e-9):
        if self.scale_type == 1:
            return 1 / (abs(scale) + eps)
        if self.scale_type == 2:
            return -scale
        raise ValueError("Scale Type %d is not supported!" % self.scale_type)

    def scale_init(self):
        if self.scale_type == 1:
            return 1.0
        if self.scale_type == 2:
            return 0.0
        raise ValueError("Scale Type %d is not supported!" % self.scale_type)

    def orth_embedding(self, embeddings, eps=1e-18, do_test=True):
        num_emb = embeddings.shape[0]
        assert embeddings.shape[1] == self.num_elem
        assert embeddings.shape[2] == (self.num_elem +
                                       (1 if self.use_scale else 0))
        if self.use_scale:
            emb_scale = embeddings[:, :, -1]
            embeddings = embeddings[:, :, :self.num_elem]

        u = [embeddings[:, 0]]
        uu = [0] * self.num_elem
        uu[0] = (u[0] * u[0]).sum(axis=-1)
        u_d = embeddings[:, 1:]
        for i in range(1, self.num_elem):
            u_d = u_d - u[-1].unsqueeze(axis=1) * (
                (embeddings[:, i:] * u[i - 1].unsqueeze(axis=1)).sum(
                    axis=-1) / uu[i - 1].unsqueeze(axis=1)).unsqueeze(-1)
            u_i = u_d[:, 0]
            if u_d.shape[1] > 1:
                u_d = u_d[:, 1:]
            uu[i] = (u_i * u_i).sum(axis=-1)
            u.append(u_i)

        u = paddle.stack(u, axis=1)  #num_emb X num_elem X num_elem
        u_norm = u.norm(axis=-1, keepdim=True, p=2)
        u = u / u_norm
        if self.use_scale:
            u = paddle.concat([u, emb_scale.unsqueeze(-1)], axis=-1)
        return u

    def orth_rel_embedding(self, relation_embedding):
        rel_emb_size = relation_embedding.shape
        ote_size = self.num_elem
        scale_dim = 1 if self.use_scale else 0
        rel_embedding = relation_embedding.reshape(
            [-1, ote_size, ote_size + scale_dim])
        rel_embedding = self.orth_embedding(rel_embedding).reshape(
            rel_emb_size)
        return rel_embedding

    def orth_reverse_mat(self, rel_embeddings):
        rel_size = rel_embeddings.shape
        if self.use_scale:
            rel_emb = rel_embeddings.reshape(
                [-1, self.num_elem, self.num_elem + 1])
            rel_mat = rel_emb[:, :, :self.num_elem].transpose([0, 2, 1])
            rel_scale = self.reverse_scale(rel_emb[:, :, self.num_elem:])
            rel_embeddings = paddle.concat(
                [rel_mat, rel_scale], axis=-1).reshape(rel_size)
        else:
            rel_embeddings = rel_embeddings.reshape(
                [-1, self.num_elem, self.num_elem]).transpose(
                    [0, 2, 1]).reshape(rel_size)
        return rel_embeddings

    #     if neg_head:
    #         relations = self.orth_reverse_mat(relations)
    #         score_result = self.score(tails, relations, heads)
    #     else:
    #         score_result = self.score(heads, relations, tails)
    #     score = self.gamma - score_result
    #     return score

    # def get_neg_score(self,
    #                   heads,
    #                   relations,
    #                   tails,
    #                   batch_size,
    #                   mini_batch_size,
    #                   neg_sample_size,
    #                   neg_head=True):
    #     if neg_head:
    #         relations = self.orth_rel_embedding(relations)
    #         relations = self.orth_reverse_mat(relations)
    #         score_result = self.neg_score(tails, relations, heads,
    #                                       neg_sample_size, mini_batch_size)
    #         score = self.gamma - score_result
    #         return score
    #     else:
    #         relations = self.orth_rel_embedding(relations)
    #         score_result = self.neg_score(heads, relations, tails,
    #                                       neg_sample_size, mini_batch_size)
    #         score = self.gamma - score_result
    #         return score
