import os
from tqdm import tqdm
import numpy as np
import pdb

def get_neighbor_list(data_path):
    # 获取实体关系字典
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
    
    # 构造train edges,由id直接构造
    with open(os.path.join(data_path, 'train.txt')) as fin:
        edges = []
        for line in fin:
            h, r, t = line.strip().split('\t')
            hid = entity2id[h]
            rid = relation2id[r]
            tid = entity2id[t]
            edges.append([hid, rid, tid])
        edges = np.array(edges, dtype=np.int32)
        
    # 构造邻居部分
    n_entities = len(entity2id)
    # 格式 一个实体的邻居：[[a, b, c], [r1, r2, r3], [1, -1, 1]](包括方向)，多个实体就用列表存起来
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
    return entity_neighbors
               
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
            re_score = re_head * re_rel + ((im_head * im_rel).T * direct).T
            im_score = im_head * re_rel - ((re_head * im_rel).T * direct).T
            src_nfeat_value = np.concatenate([re_score, im_score], -1)
            
            aggr_nfeat = np.mean(src_nfeat_value, axis=0)
            new_nfeat = entity_feat[i] * alpha + aggr_nfeat * (1 - alpha)
        else:
            new_nfeat = entity_feat[i]
        new_entity_feat[i] = new_nfeat
    return new_entity_feat
        
def ppnp_ote(entity_feat_path, relation_feat_path, entity_neighbors, alpha, gamma=10):
    new_entity_feat = np.zeros(entity_feat.shape, dtype="float32")
    

# For FB15k-237 and WN18RR.
def main_small(model, dataset, alpha=0.98, k_hop=10, gamma=10):
    data_path = os.path.join('/home/work/suweiyue/Shared/WikiKG90M/RPS/KGEmbedding-OTE/data/', dataset)
    feat_path = os.path.join('/home/work/suweiyue/Shared/WikiKG90M/RPS/KGEmbedding-OTE/models', '%s_%s' % (model, dataset))
    entity_feat_path = os.path.join(feat_path, "best/entity_embedding.npy")
    relation_feat_path = os.path.join(feat_path, "best/relation_embedding.npy")
    entity_feat = np.load(entity_feat_path)
    relation_feat = np.load(relation_feat_path)
    entity_neighbors = get_neighbor_list(data_path)
    if model == 'TransE':
        for i in range(k_hop):
            entity_feat = ppnp_transe(entity_feat, relation_feat, entity_neighbors, alpha=alpha)
    elif model == 'RotatE':
        for i in range(k_hop):
            entity_feat = ppnp_rotate(entity_feat, relation_feat, entity_neighbors, alpha=alpha)
    elif model == 'OTE':
        for i in range(k_hop):
            entity_feat = ppnp_ote(entity_feat, relation_feat, entity_neighbors, alpha=alpha)
    # 保存appnp后的embedding，文件夹名称为 appnp_alpha-0.8_khop_10
    appnp_feat_path = os.path.join(feat_path, "appnp_feat")
    if not os.path.exists(appnp_feat_path):
        os.mkdir(appnp_feat_path)
    save_path = os.path.join(appnp_feat_path, "alpha_%f-khop_%d" % (alpha, k_hop))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, "entity_embedding.npy"), entity_feat)
    os.system("cp %s %s" % (os.path.join(feat_path, "relation_embedding.npy"), save_path))
        
# main_small('TransE', 'FB15k-237')
main_small('RotatE', 'FB15k-237')

