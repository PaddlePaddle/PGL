import os
import argparse
from tqdm import tqdm
import numpy as np
import pdb
import torch
from evaluate import evaluate_test # currently not uploaded
from ote_orth import Orth_OTE      # currently not uploaded


def get_neighbor_list(data_path):
    
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
    # 格式 实体的邻居：[[a, b, c], [r1, r2, r3], [1, -1, 1]](包括方向)，多个实体就用列表存起来
    entity_neighbors = []
    for eid in tqdm(range(n_entities)):
        nb = [[], [], []]
        # eid位于头实体
        h_index = np.where(edges[:, 0] == eid)
        nb[0].extend(list(edges[h_index[0]][:, 2]))
        nb[1].extend(list(edges[h_index[0]][:, 1]))
        nb[2].extend([-1] * len(h_index[0]))
        # eid位于尾实体
        t_index = np.where(edges[:, 2] == eid) 
        nb[0].extend(list(edges[t_index[0]][:, 0]))
        nb[1].extend(list(edges[t_index[0]][:, 1]))
        nb[2].extend([1] * len(t_index[0]))
        entity_neighbors.append(nb)
    return entity_neighbors, edges

           
def ppnp_transe(entity_feat, relation_feat, entity_neighbors, alpha): 
    new_entity_feat = np.zeros(entity_feat.shape, dtype="float32")
    for i, efeat in tqdm(enumerate(entity_feat)):
        src_or_dst, r_type, direct = entity_neighbors[i]
        if len(src_or_dst) > 0:
            src_nfeat_value = entity_feat[src_or_dst]
            neigh_nfeat = src_nfeat_value + (relation_feat[r_type].T * direct).T
            aggr_nfeat = np.mean(neigh_nfeat, axis=0)
            new_nfeat = entity_feat[i] * alpha + aggr_nfeat * (1 - alpha)
        else:
            new_nfeat = entity_feat[i]
        new_entity_feat[i] = new_nfeat
    return new_entity_feat


def ppnp_rotate(entity_feat, relation_feat, entity_neighbors, alpha, gamma=10):
    new_entity_feat = np.zeros(entity_feat.shape, dtype="float32")
    
    hidden_dim = relation_feat.shape[1]
    emb_init = gamma / hidden_dim
    for i, efeat in tqdm(enumerate(entity_feat)):
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
            aggr_nfeat = np.mean(src_nfeat_value, axis=0)
            new_nfeat = entity_feat[i] * alpha + aggr_nfeat * (1 - alpha)
        else:
            new_nfeat = entity_feat[i]
        new_entity_feat[i] = new_nfeat
    return new_entity_feat


def ppnp_ote(entity_feat, relation_feat, entity_neighbors, alpha, 
             gamma=10, r_emb=None, r_emb_mat=None, ote_size=20):
    
    new_entity_feat = np.zeros(entity_feat.shape, dtype="float32")
    for i, efeat in tqdm(enumerate(entity_feat)):
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
            rel = inputs_rel.reshape(-1, ote_size ,  ote_size + 1)
            scale = np.exp(rel[:,:,ote_size:])
            scale = scale / np.linalg.norm(scale, axis=-1, keepdims=True)
            rel_scale = rel[:,:,:ote_size] * scale 
            outputs = np.matmul(inputs, rel_scale)
            outputs = outputs.reshape(inputs_size)
            tmp_nfeat_list[j] = outputs
            
        if len(tmp_nfeat_list) > 0:
            neigh_nfeat = tmp_nfeat_list
            aggr_nfeat = np.mean(neigh_nfeat, axis=0)
            new_nfeat = efeat * alpha + aggr_nfeat * (1 - alpha)
        else:
            new_nfeat = efeat
        new_entity_feat[i] = new_nfeat
    return new_entity_feat


def ppnp_relation(model, entity_feat, relation_feat, edges, alpha):
    new_relation_feat = np.zeros(relation_feat.shape, dtype="float32")
    if model == 'TransE':
        for i, rfeat in enumerate(relation_feat):
            i_edge = np.where(edges[:, 1] == i)
            tail = edges[i_edge[0]][:, 2]
            head = edges[i_edge[0]][:, 0]
            tail_emb = entity_feat[tail]
            head_emb = entity_feat[head]
            aggr_rfeat = np.mean(tail_emb - head_emb, axis=0)
            new_rfeat = rfeat * alpha + aggr_rfeat * (1 - alpha)
            new_relation_feat[i] = new_rfeat
    elif model == 'RotatE':
        pass
    elif model == 'OTE':
        pass
    
    return new_relation_feat
            
    
def main(model_name, dataset, edges, entity_feat, relation_feat, entity_neighbors, 
         alpha=0.98, k_hop=10, gamma=6.0, r_emb=None, r_emb_mat=None, ote_size=20,
         relation_appnp=False):
    """For small datasets like FB15k-237 and wn18rr.
    """
    
    for i in range(k_hop):
        if model_name == 'TransE':
            entity_feat = ppnp_transe(entity_feat, relation_feat, entity_neighbors, alpha=alpha)
            if relation_appnp:
                relation_feat = ppnp_relation(model_name, entity_feat, relation_feat, edges, alpha)
        elif model_name == 'RotatE':
            entity_feat = ppnp_rotate(entity_feat, relation_feat, entity_neighbors, alpha=alpha)
        elif model_name == 'OTE':
            entity_feat = ppnp_ote(entity_feat, relation_feat, entity_neighbors, 
                                   alpha=alpha, r_emb=r_emb, r_emb_mat=r_emb_mat,
                                   ote_size=ote_size)
        appnp_feat_path = os.path.join(feat_path, "appnp_feat")
        if not os.path.exists(appnp_feat_path):
            os.mkdir(appnp_feat_path)
        if relation_appnp:
            save_path = os.path.join(appnp_feat_path, "relation_alpha_%f-khop_%d" % (alpha, i + 1))
        else:
            save_path = os.path.join(appnp_feat_path, "alpha_%f-khop_%d" % (alpha, i + 1))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(os.path.join(save_path, "entity_embedding.npy"), entity_feat)
        np.save(os.path.join(save_path, "relation_embedding.npy"), relation_feat)
        
        evaluate_test(model_name, dataset, save_path)
        
    
if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser("APPNP")
    parser.add_argument(
        "--data_path", type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, default="FB15k-237", help="dataset (FB15k-237, wn18rr)")
    parser.add_argument(
        "--model_path", type=str, default=None)
    parser.add_argument(
        "--model_name", type=str, default='TransE', help="model (TransE, RotatE, OTE)")
    parser.add_argument(
        "--khop", type=int, default=10, help='APPNP k hop')
    parser.add_argument(
        "--gamma", type=float, default=10, help='used in RotatE')
    parser.add_argument(
        "--ote_size", type=int, default=20, help='used in OTE')
    parser.add_argument(
        "--relation_appnp", action="store_true")
    args = parser.parse_args()
    
    data_path = os.path.join(args.data_path, args.dataset)
    feat_path = os.path.join(args.model_path, '%s_%s' % (args.model_name, args.dataset))
    entity_feat_path = os.path.join(feat_path, "best/entity_embedding.npy")
    relation_feat_path = os.path.join(feat_path, "best/relation_embedding.npy")
    
    if args.model_name == 'OTE':
        ote = Orth_OTE(relation_feat_path, args.ote_size)
        r_emb = ote.orth_relation_emb
        r_emb_mat = ote.orth_relation_emb_mat
    else:
        r_emb = None
        r_emb_mat = None
        
    entity_feat = np.load(entity_feat_path)
    relation_feat = np.load(relation_feat_path)
    entity_neighbors, edges = get_neighbor_list(data_path)
    
    get_indegree(len(entity_neighbors), edges)
    
    '''
    for alpha in [0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.998]: # TODO: alpha改为衰减学习率 
        main(args.model_name, args.dataset, edges, entity_feat, relation_feat, 
             entity_neighbors, alpha=alpha, k_hop=args.khop, 
             gamma=args.gamma, r_emb=r_emb, r_emb_mat=r_emb_mat, 
             ote_size=args.ote_size, 
             relation_appnp=args.relation_appnp)
    '''
