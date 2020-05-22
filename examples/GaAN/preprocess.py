"""
将 ogb_proteins 的数据处理为 PGL 的 graph 数据，并返回 graph, label, train/valid/test 等信息
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from ogb.nodeproppred import NodePropPredDataset, Evaluator

import pgl
import numpy as np
import os
import time


def get_graph_data(d_name="ogbn-proteins", mini_data=False):
    """
        Param:
            d_name: name of dataset
            mini_data: if mini_data==True, only use a small dataset (for test)
    """
    # 导入 ogb 数据
    dataset = NodePropPredDataset(name = d_name)
    num_tasks = dataset.num_tasks # obtaining the number of prediction tasks in a dataset

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0]
    
    # 调整维度，符合 PGL 的 Graph 要求
    graph["edge_index"] = graph["edge_index"].T
    
    # 使用小规模数据，500个节点
    if mini_data: 
        graph['num_nodes'] = 500
        mask = (graph['edge_index'][:, 0] < 500)*(graph['edge_index'][:, 1] < 500)
        graph["edge_index"] = graph["edge_index"][mask]
        graph["edge_feat"] = graph["edge_feat"][mask]
        label = label[:500]
        train_idx = np.arange(0,400)
        valid_idx = np.arange(400,450)
        test_idx = np.arange(450,500)
    
    # 输出 dataset 的信息    
    print(graph.keys())
    print("节点个数 ", graph["num_nodes"])
    print("节点最小编号", graph['edge_index'][0].min())
    print("边个数 ", graph["edge_index"].shape[1])
    print("边索引 shape ", graph["edge_index"].shape)
    print("边特征 shape ", graph["edge_feat"].shape)
    print("节点特征是 ", graph["node_feat"])
    print("species shape", graph['species'].shape)
    print("label shape ", label.shape)
    
    # 读取/计算 node feature
    # 确定读取文件的路径
    if mini_data:
        node_feat_path = './dataset/ogbn_proteins_node_feat_small.npy'
    else:
        node_feat_path = './dataset/ogbn_proteins_node_feat.npy'

    new_node_feat = None
    if os.path.exists(node_feat_path):
        # 如果文件存在，直接读取
        print("读取 node feature 开始".center(50, '='))
        new_node_feat = np.load(node_feat_path)
        print("读取 node feature 成功".center(50, '='))
    else:
        # 如果文件不存在，则计算
        # 每个节点 i 的特征为其邻边特征的均值
        print("计算 node feature 开始".center(50, '='))
        start = time.perf_counter()
        for i in range(graph['num_nodes']):
            if i % 100 == 0:
                dur = time.perf_counter() - start
                print("{}/{}({}%), times: {:.2f}s".format(
                    i, graph['num_nodes'], i/graph['num_nodes']*100, dur
                ))
            mask = (graph['edge_index'][:, 0] == i) # 选择 i 的所有邻边
            # 计算均值
            current_node_feat = np.mean(np.compress(mask, graph['edge_feat'], axis=0),
                                        axis=0, keepdims=True)
            if i == 0:
                new_node_feat = [current_node_feat]
            else:  
                new_node_feat.append(current_node_feat)

        new_node_feat = np.concatenate(new_node_feat, axis=0)
        print("计算 node feature 结束".center(50,'='))

        print("存储 node feature 中，在"+node_feat_path.center(50, '='))
        np.save(node_feat_path, new_node_feat)
        print("存储 node feature 结束".center(50,'='))
    
    print(new_node_feat)
    
    
    # 构造 Graph 对象
    g = pgl.graph.Graph(
        num_nodes=graph["num_nodes"],
        edges = graph["edge_index"],
        node_feat = {'node_feat': new_node_feat},
        edge_feat = None
    )
    print("创建 Graph 对象成功")
    print(g)
    return g, label, train_idx, valid_idx, test_idx, Evaluator(d_name)
    