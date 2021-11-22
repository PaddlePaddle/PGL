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
from numpy.random import RandomState
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.functional import log_sigmoid


class ScoreFunc(object):
    """Abstract implementation of score function.
    """

    def __init__(self):
        super(ScoreFunc, self).__init__()
        self.embed_epsilon = 2.0

    def __call__(self, head, rel, tail):
        raise NotImplementedError('foward of ScoreFunc not implemented!')

    def get_neg_score(self, head, rel, tail, neg_head=False):
        """Compute scores for negative samples.
        """
        raise NotImplementedError(
            'get_neg_score of ScoreFunc not implemented!')

    def get_er_regularization(self, ent_embeds, rel_embeds, args):
        """Compute regularization of entity and relation embeddings.
        """
        value = paddle.sum(ent_embeds.abs().pow(args.reg_norm)) + \
            paddle.sum(rel_embeds.abs().pow(args.reg_norm))
        value = args.reg_coef * value
        return value

    def get_hrt_regularization(self, head, rel, tail, args):
        """Compute regularization of heads, relations and tails.
        """
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

        value = args.quate_lmbda1 * reg_ents + args.quate_lmbda2 * reg_rels

        return value


class TransEScore(ScoreFunc):
    """
    Translating embeddings for modeling multi-relational data.
    https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf
    """

    def __init__(self, gamma):
        super(TransEScore, self).__init__()
        self.gamma = gamma

    def __call__(self, head, rel, tail):
        head = head + rel
        score = self.gamma - paddle.norm(head - tail, p=2, axis=-1)
        return score

    def get_neg_score(self, head, rel, tail, neg_head=False):
        if neg_head:
            tail = tail - rel
            score = self.gamma - self.cdist(tail, head)
        else:
            head = head + rel
            score = self.gamma - self.cdist(head, tail)
        return score

    def cdist(self, a, b):
        """Euclidean distance.
        """
        a_s = paddle.norm(a, p=2, axis=-1).pow(2)
        b_s = paddle.norm(b, p=2, axis=-1).pow(2)
        dist_score = -2 * paddle.bmm(a, b.transpose([0, 2, 1]))
        dist_score = dist_score + b_s.unsqueeze(-2) + a_s.unsqueeze(-1)
        dist_score = paddle.sqrt(paddle.clip(dist_score, min=1e-30))
        return dist_score


class DistMultScore(ScoreFunc):
    """
    Embedding Entities and Relations for Learning and Inference in Knowledge Bases.
    https://arxiv.org/abs/1412.6575
    """

    def __init__(self):
        super(DistMultScore, self).__init__()

    def __call__(self, head, rel, tail):
        score = head * rel * tail
        score = paddle.sum(score, axis=-1)
        return score

    def get_neg_score(self, head, rel, tail, neg_head=False):
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
        self.emb_init = self._get_init_weight(embed_dim)

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

    def _get_init_weight(self, embed_dim):
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
    """
    Complex Embeddings for Simple Link Prediction.
    https://arxiv.org/abs/1606.06357
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
    """
    Quaternion Knowledge Graph Embedding.
    https://arxiv.org/abs/1904.10281
    """

    def __init__(self, num_ents):
        super(QuatEScore, self).__init__()
        self.num_ents = num_ents

    def __call__(self, head, rel, tail):
        A, B, C, D = self._get_part_score(head, rel)
        tails = paddle.chunk(tail, chunks=4, axis=-1)

        score = (A * tails[0] + B * tails[1] + C * tails[2] + D * tails[3])
        score = paddle.sum(score, axis=-1)
        return -score

    def get_neg_score(self, head, rel, tail, neg_head=False):
        if neg_head:
            return self._get_neg_score(tail, rel, head)
        else:
            return self._get_neg_score(head, rel, tail)

    def _get_part_score(self, a, b):
        a = paddle.chunk(a, chunks=4, axis=-1)
        b = paddle.chunk(b, chunks=4, axis=-1)

        denominator_b = paddle.sqrt(b[0]**2 + b[1]**2 + b[2]**2 + b[3]**2 +
                                    1e-10)
        b[0] = b[0] / denominator_b
        b[1] = b[1] / denominator_b
        b[2] = b[2] / denominator_b
        b[3] = b[3] / denominator_b

        A = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        B = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
        C = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
        D = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]

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
        a, b, c, d = paddle.chunk(neg_e, chunks=4, axis=-1)
        a = a.transpose((0, 2, 1))
        b = b.transpose((0, 2, 1))
        c = c.transpose((0, 2, 1))
        d = d.transpose((0, 2, 1))

        score = paddle.bmm(A, a) + paddle.bmm(B, b) + \
                paddle.bmm(C, c) + paddle.bmm(D, d)
        return -score


class OTEScore(ScoreFunc):
    """
    Orthogonal Relation Transforms with Graph Context Modeling for Knowledge Graph Embedding.
    https://aclanthology.org/2020.acl-main.241/
    """

    def __init__(self, gamma, num_elem, scale_type=0):
        super(OTEScore, self).__init__()
        self.gamma = gamma
        self.num_elem = num_elem
        self.scale_type = scale_type

    @property
    def use_scale(self):
        """Return True if use scale for relations.
        """
        return self.scale_type > 0

    @property
    def scale_init(self):
        """Return initial value of scales.
        """
        if self.scale_type == 1:
            return 1.0
        if self.scale_type == 2:
            return 0.0
        raise ValueError("Scale Type %d is not supported!" % self.scale_type)

    def get_scale(self, scale):
        """Get scaled tensor.
        """
        if self.scale_type == 1:
            return scale.abs()
        if self.scale_type == 2:
            return scale.exp()
        raise ValueError("Scale Type %d is not supported!" % self.scale_type)

    def reverse_scale(self, scale, eps=1e-9):
        """Get scaled tensor of inverse relations.
        """
        if self.scale_type == 1:
            return 1 / (scale.abs() + eps)
        if self.scale_type == 2:
            return -scale
        raise ValueError("Scale Type %d is not supported!" % self.scale_type)

    def __call__(self, head, rel, tail):
        rel = self.orth_rel_embedding(rel)

        assert head.shape[:-1] == rel.shape[:-1]
        shape = head.shape
        head = head.reshape([-1, 1, self.num_elem])
        if self.use_scale:
            rel = rel.reshape([-1, self.num_elem, self.num_elem + 1])
            scale = self.get_scale(rel[:, :, self.num_elem:])
            scale = scale / scale.norm(axis=-1, p=2, keepdim=True)
            rel = rel[:, :, :self.num_elem] * scale
        else:
            rel = rel.reshape([-1, self.num_elem, self.num_elem])
        output = paddle.bmm(head, rel).reshape(shape)
        score = self._get_score(output, tail)
        return self.gamma - score

    def get_neg_score(self, head, rel, tail, neg_head=False):
        """Calculate scores of negative samples.
        """
        rel = self.orth_rel_embedding(rel)
        if neg_head:
            rel = self._orth_reverse_mat(rel)
            score = self._get_neg_score(tail, rel, head)
        else:
            score = self._get_neg_score(head, rel, tail)
        return self.gamma - score

    def _get_neg_score(self, pos_e, pos_r, neg_e):
        chunk_size = pos_e.shape[1]
        neg_sample_size = neg_e.shape[1]
        embed_dim = pos_e.shape[-1]

        assert pos_e.shape[:-1] == pos_r.shape[:-1]
        pos_e = pos_e.reshape([-1, 1, self.num_elem])
        if self.use_scale:
            pos_r = pos_r.reshape([-1, self.num_elem, self.num_elem + 1])
            scale = self.get_scale(pos_r[:, :, self.num_elem:])
            scale = scale / scale.norm(axis=-1, p=2, keepdim=True)
            scale_r = pos_r[:, :, :self.num_elem] * scale
            output = paddle.bmm(pos_e, scale_r)
        else:
            pos_r = pos_r.reshape([-1, self.num_elem, self.num_elem])
            output = paddle.bmm(pos_e, pos_r)

        output = output.reshape([-1, chunk_size, 1, embed_dim])
        neg_e = neg_e.reshape([-1, 1, neg_sample_size, embed_dim])

        score = self._get_score(output, neg_e)
        return score

    def _get_score(self, combined_embeds, ent_embeds):
        output = combined_embeds - ent_embeds
        output_shape = output.shape
        embed_dim = output_shape[-1]
        output = output.reshape([-1, self.num_elem]).norm(p=2, axis=-1)
        output = output.reshape([-1, embed_dim // self.num_elem]).sum(axis=-1)
        score = output.reshape(output_shape[:-1])
        return score

    def gram_schimidt_process(self, embeds, eps=1e-18, do_test=True):
        """ Orthogonalize embeddings.
        """
        assert embeds.shape[1] == self.num_elem
        assert embeds.shape[2] == (self.num_elem + int(self.use_scale))
        if self.use_scale:
            scales = embeds[:, :, -1]
            embeds = embeds[:, :, :self.num_elem]

        u = [embeds[:, 0]]
        uu = [0] * self.num_elem
        uu[0] = (u[0] * u[0]).sum(axis=-1)
        u_d = embeds[:, 1:]
        for i in range(1, self.num_elem):
            tmp = (embeds[:, i:] * u[i - 1].unsqueeze(axis=1)).sum(axis=-1)
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
            u = paddle.concat([u, scales.unsqueeze(-1)], axis=-1)
        return u

    def orth_rel_embedding(self, embeds):
        """Orthogonalize relation embeddings.
        """
        embed_shape = embeds.shape
        embeds = embeds.reshape(
            [-1, self.num_elem, self.num_elem + int(self.use_scale)])
        embeds = self.gram_schimidt_process(embeds)
        embeds = embeds.reshape(embed_shape)
        return embeds

    def _orth_reverse_mat(self, embeds):
        """Transpose the orthogonalized relation embeddings.
        """
        embed_shape = embeds.shape
        if self.use_scale:
            embeds = embeds.reshape([-1, self.num_elem, self.num_elem + 1])
            rel_mat = embeds[:, :, :self.num_elem].transpose([0, 2, 1])
            rel_scale = self.reverse_scale(embeds[:, :, self.num_elem:])
            embeds = paddle.concat(
                [rel_mat, rel_scale], axis=-1).reshape(embed_shape)
        else:
            embeds = embeds.reshape([-1, self.num_elem, self.num_elem])
            embeds = embeds.transpose([0, 2, 1]).reshape(embed_shape)
        return embeds
