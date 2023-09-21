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

import os
import time
import logging
import argparse

import numpy as np
from tqdm import tqdm
import pgl
from ote_orth import OrthOTE

logging.basicConfig(format='', level=logging.INFO)


def get_neighbor_list_wikikg2():
    from ogb.linkproppred import LinkPropPredDataset
    dataset = LinkPropPredDataset(name="ogbl-wikikg2")
    split_edge = dataset.get_edge_split()
    train_edges = split_edge["train"]

    head_id = train_edges["head"].reshape((-1, 1))
    tail_id = train_edges["tail"].reshape((-1, 1))
    relation_id = train_edges["relation"]

    edges = np.concatenate([head_id, tail_id], axis=-1)
    n_entities = 2500604
    graph = pgl.Graph(
        num_nodes=n_entities,
        edges=edges,
        edge_feat={"edge_feature": relation_id})

    entity_neighbors = []
    for nid in tqdm(range(n_entities)):
        nb = [[], [], []]
        # nid-head
        succ, succ_eid = graph.successor([nid], return_eids=True)
        nb[0].extend(list(succ[0]))
        nb[1].extend(list(graph.edge_feat["edge_feature"][list(succ_eid[0])]))
        nb[2].extend([-1] * len(succ[0]))

        # nid-tail
        pred, pred_eid = graph.predecessor([nid], return_eids=True)
        nb[0].extend(list(pred[0]))
        nb[1].extend(list(graph.edge_feat["edge_feature"][list(pred_eid[0])]))
        nb[2].extend([1] * len(pred[0]))
        entity_neighbors.append(nb)
    return entity_neighbors, None


def get_neighbor_list_fb_wn(data_path):
    with open(os.path.join(data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    with open(os.path.join(data_path, 'train.txt')) as fin:
        edges = []
        for line in fin:
            h, r, t = line.strip().split('\t')
            hid = entity2id[h]
            rid = relation2id[r]
            tid = entity2id[t]
            edges.append([hid, rid, tid])
        edges = np.array(edges, dtype=np.int32)

    n_entities = len(entity2id)
    entity_neighbors = []
    for eid in range(n_entities):
        nb = [[], [], []]
        h_index = np.where(edges[:, 0] == eid)
        nb[0].extend(list(edges[h_index[0]][:, 2]))
        nb[1].extend(list(edges[h_index[0]][:, 1]))
        nb[2].extend([-1] * len(h_index[0]))
        t_index = np.where(edges[:, 2] == eid)
        nb[0].extend(list(edges[t_index[0]][:, 0]))
        nb[1].extend(list(edges[t_index[0]][:, 1]))
        nb[2].extend([1] * len(t_index[0]))
        entity_neighbors.append(nb)
    return entity_neighbors, edges


def get_indegree(n_entities, edges):
    indegrees = np.zeros(n_entities)
    for eid in range(n_entities):
        h_index = np.where(edges[:, 2] == eid)
        indegrees[eid] = len(h_index[0]) + 1
    indegrees = indegrees.reshape((-1, 1))
    return indegrees


def rep_transe(entity_feat,
               relation_feat,
               entity_neighbors,
               alpha,
               degree_w,
               indegrees=None,
               neighbor_norm=False):
    new_entity_feat = np.zeros(entity_feat.shape, dtype="float32")
    for i, efeat in enumerate(entity_feat):
        src_or_dst, r_type, direct = entity_neighbors[i]
        if len(src_or_dst) > 0:
            src_nfeat_value = entity_feat[src_or_dst]
            neigh_nfeat = src_nfeat_value + (relation_feat[r_type].T * direct
                                             ).T

            if not neighbor_norm:
                aggr_nfeat = np.mean(neigh_nfeat, axis=0)
            else:
                src_indegrees = indegrees[src_or_dst]
                src_norm = np.power(src_indegrees, degree_w)
                src_norm = src_norm / np.sum(src_norm)
                neigh_nfeat = neigh_nfeat * src_norm
                aggr_nfeat = np.sum(neigh_nfeat, axis=0)
            new_nfeat = efeat * alpha + aggr_nfeat * (1 - alpha)
        else:
            new_nfeat = efeat
        new_entity_feat[i] = new_nfeat
    return new_entity_feat


def rep_rotate(entity_feat,
               relation_feat,
               entity_neighbors,
               alpha,
               degree_w,
               gamma=10,
               indegrees=None,
               neighbor_norm=False):
    new_entity_feat = np.zeros(entity_feat.shape, dtype="float32")

    hidden_dim = relation_feat.shape[1]
    emb_init = gamma / hidden_dim

    for i, efeat in enumerate(entity_feat):
        src_or_dst, r_type, direct = entity_neighbors[i]
        if len(src_or_dst) > 0:
            relation = relation_feat[r_type]
            phase_rel = relation / (emb_init / np.pi)
            re_rel, im_rel = np.cos(phase_rel), np.sin(phase_rel)

            head = entity_feat[src_or_dst]
            re_head, im_head = np.split(head, 2, -1)
            re_score = re_head * re_rel - ((im_head * im_rel).T * direct).T
            im_score = im_head * re_rel + ((re_head * im_rel).T * direct).T
            src_nfeat_value = np.concatenate([re_score, im_score], -1)

            if not neighbor_norm:
                aggr_nfeat = np.mean(src_nfeat_value, axis=0)
            else:
                src_indegrees = indegrees[src_or_dst]
                src_norm = np.power(src_indegrees, degree_w)
                src_norm = src_norm / np.sum(src_norm)
                neigh_nfeat = neigh_nfeat * src_norm
                aggr_nfeat = np.sum(neigh_nfeat, axis=0)
            new_nfeat = efeat * alpha + aggr_nfeat * (1 - alpha)
        else:
            new_nfeat = efeat
        new_entity_feat[i] = new_nfeat
    return new_entity_feat


def rep_distmult(entity_feat,
                 relation_feat,
                 entity_neighbors,
                 alpha,
                 degree_w,
                 indegrees=None,
                 neighbor_norm=False):
    new_entity_feat = np.zeros(entity_feat.shape, dtype="float32")
    for i, efeat in enumerate(entity_feat):
        src_or_dst, r_type, direct = entity_neighbors[i]
        if len(src_or_dst) > 0:
            src_nfeat_value = entity_feat[src_or_dst]
            neigh_nfeat = relation_feat[r_type] * src_nfeat_value

            if not neighbor_norm:
                aggr_nfeat = np.mean(neigh_nfeat, axis=0)
            else:
                src_indegrees = indegrees[src_or_dst]
                src_norm = np.power(src_indegrees, degree_w)
                src_norm = src_norm / np.sum(src_norm)
                neigh_nfeat = neigh_nfeat * src_norm
                aggr_nfeat = np.sum(neigh_nfeat, axis=0)
            new_nfeat = efeat * alpha + aggr_nfeat * (1 - alpha)
        else:
            new_nfeat = efeat
        new_entity_feat[i] = new_nfeat
    return new_entity_feat


def rep_ote(entity_feat,
            relation_feat,
            entity_neighbors,
            alpha,
            degree_w,
            r_emb=None,
            r_emb_mat=None,
            ote_size=20,
            indegrees=None,
            neighbor_norm=False,
            scale_norm=False):
    """For OTE and GC-OTE."""
    new_entity_feat = np.zeros(entity_feat.shape, dtype="float32")
    for i, efeat in enumerate(entity_feat):
        src_or_dst, r_type, direct = entity_neighbors[i]
        src_nfeat_value = entity_feat[src_or_dst]
        tmp_nfeat_list = np.zeros((len(src_or_dst), entity_feat.shape[1]))
        for j, nfeat in enumerate(src_nfeat_value):
            if direct[j] == 1:
                inputs_rel = r_emb[r_type[j]]
            elif direct[j] == -1:
                inputs_rel = r_emb_mat[r_type[j]]
            else:
                raise ValueError
            inputs_size = nfeat.shape
            inputs = nfeat.reshape(-1, 1, ote_size)
            rel = inputs_rel.reshape(-1, ote_size, ote_size + 1)
            scale = np.exp(rel[:, :, ote_size:])
            if scale_norm:
                scale = scale / np.linalg.norm(scale, axis=-1, keepdims=True)
            rel_scale = rel[:, :, :ote_size] * scale
            outputs = np.matmul(inputs, rel_scale)
            outputs = outputs.reshape(inputs_size)
            tmp_nfeat_list[j] = outputs

        if len(tmp_nfeat_list) > 0:
            neigh_nfeat = tmp_nfeat_list
            if not neighbor_norm:
                aggr_nfeat = np.mean(neigh_nfeat, axis=0)
            else:
                src_indegrees = indegrees[src_or_dst]
                src_norm = np.power(src_indegrees, degree_w)
                src_norm = src_norm / np.sum(src_norm)
                neigh_nfeat = neigh_nfeat * src_norm
                aggr_nfeat = np.sum(neigh_nfeat, axis=0)
            new_nfeat = efeat * alpha + aggr_nfeat * (1 - alpha)
        else:
            new_nfeat = efeat
        new_entity_feat[i] = new_nfeat
    return new_entity_feat


def main(model_name,
         dataset,
         entity_feat,
         relation_feat,
         entity_neighbors,
         alpha=0.98,
         k_hop=10,
         gamma=6.0,
         degree_w=0.1,
         r_emb=None,
         r_emb_mat=None,
         ote_size=20,
         indegrees=None,
         neighbor_norm=False,
         scale_norm=False):

    for i in range(k_hop):
        start = time.time()
        if model_name == 'TransE':
            entity_feat = rep_transe(
                entity_feat,
                relation_feat,
                entity_neighbors,
                alpha=alpha,
                degree_w=degree_w,
                indegrees=indegrees,
                neighbor_norm=neighbor_norm)
        elif model_name == 'RotatE':
            entity_feat = rep_rotate(
                entity_feat,
                relation_feat,
                entity_neighbors,
                alpha=alpha,
                degree_w=degree_w,
                gamma=gamma,
                indegrees=indegrees,
                neighbor_norm=neighbor_norm)
        elif model_name == 'OTE':
            entity_feat = rep_ote(
                entity_feat,
                relation_feat,
                entity_neighbors,
                alpha=alpha,
                degree_w=degree_w,
                r_emb=r_emb,
                r_emb_mat=r_emb_mat,
                ote_size=ote_size,
                indegrees=indegrees,
                neighbor_norm=neighbor_norm,
                scale_norm=scale_norm)
        elif model_name == 'DistMult':
            entity_feat = rep_distmult(
                entity_feat,
                relation_feat,
                entity_neighbors,
                alpha=alpha,
                degree_w=degree_w,
                indegrees=indegrees,
                neighbor_norm=neighbor_norm)
        end = time.time()
        print("Time elapsed for running one hop: %.4f" % (end - start))
    save_path = "REP_save_feat_%s_%s" % (model_name, dataset)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        np.save(os.path.join(save_path, "entity_embedding.npy"), entity_feat)
        np.save(
            os.path.join(save_path, "relation_embedding.npy"), relation_feat)

    # Then you can use the saved embeddings to get new evaluation results.


if __name__ == "__main__":
    parser = argparse.ArgumentParser("REP")
    parser.add_argument(
        "--dataset",
        type=str,
        default="FB15k-237",
        help="Dataset (FB15k-237, wn18rr, wikikg2)")
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="The data path for FB15k-237 and wn18rr.")
    parser.add_argument(
        "--model_name",
        type=str,
        default='TransE',
        help="model (TransE, RotatE, DistMult, OTE)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="The embedding path of different models.")

    parser.add_argument("--khop", type=int, default=20, help="REP K hops.")
    parser.add_argument(
        "--alpha",
        default=0.98,
        type=float,
        help="Hyperparameter used in REP.")
    parser.add_argument(
        "--gamma",
        type=float,
        default=10,
        help="hyperparameter used in RotatE, "
        "which should be same in both training phase and REP phase.")
    parser.add_argument(
        "--ote_size",
        type=int,
        default=20,
        help="Hyperparameter used in OTE and GC-OTE, "
        "which should be same in both training phase and REP phase.")
    parser.add_argument(
        "--degree_w",
        type=float,
        default=0.1,
        help="hyperparameter for neighbor_norm")
    parser.add_argument("--neighbor_norm", action="store_true")
    parser.add_argument(
        "--scale_norm", action="store_true", help="used in OTE")

    args = parser.parse_args()
    logging.info(args)

    entity_feat_path = os.path.join(args.model_path, "entity_embedding.npy")
    relation_feat_path = os.path.join(args.model_path,
                                      "relation_embedding.npy")
    if args.model_name in ['OTE', 'GC_OTE']:
        ote = OrthOTE(relation_feat_path, args.ote_size)
        r_emb = ote.orth_relation_emb.numpy()
        r_emb_mat = ote.orth_relation_emb_mat.numpy()
    else:
        r_emb = None
        r_emb_mat = None

    entity_feat = np.load(entity_feat_path)
    relation_feat = np.load(relation_feat_path)
    if args.dataset in ['FB15k-237', 'wn18rr']:
        entity_neighbors, edges = get_neighbor_list_fb_wn(args.data_path)
    if args.dataset in ['wikikg2']:
        entity_neighbors, edges = get_neighbor_list_wikikg2()

    if args.neighbor_norm:
        assert (args.dataset not in ['wikikg2'])
        indegrees = get_indegree(len(entity_neighbors), edges)
    else:
        indegrees = None

    main(
        args.model_name,
        args.dataset,
        entity_feat,
        relation_feat,
        entity_neighbors,
        alpha=args.alpha,
        k_hop=args.khop,
        gamma=args.gamma,
        degree_w=args.degree_w,
        r_emb=r_emb,
        r_emb_mat=r_emb_mat,
        ote_size=args.ote_size,
        indegrees=indegrees,
        neighbor_norm=args.neighbor_norm,
        scale_norm=args.scale_norm)
