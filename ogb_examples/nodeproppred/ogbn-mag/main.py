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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CPU_NUM'] = str(20)
import numpy as np
import copy
import paddle
import paddle.fluid as fluid
import pgl

#from pgl.sample import graph_saint_random_walk_sample
from pgl.sample import deepwalk_sample
from pgl.contrib.ogb.nodeproppred.dataset_pgl import PglNodePropPredDataset
from rgcn import RGCNModel, softmax_loss, paper_mask
from pgl.utils.mp_mapper import mp_reader_mapper
from pgl.utils.share_numpy import ToShareMemGraph


def hetero2homo(heterograph):
    edge = []
    for edge_type in heterograph.edge_types_info():
        edge.append(heterograph[edge_type].edges)
    edges = np.vstack(edge)
    g = pgl.graph.Graph(num_nodes=heterograph.num_nodes, edges=edges)
    g.outdegree()
    ToShareMemGraph(g)
    return g


def extract_edges_from_nodes(hetergraph, sample_nodes):
    eids = {}
    for key in hetergraph.edge_types_info():
        graph = hetergraph[key]
        eids[key] = pgl.graph_kernel.extract_edges_from_nodes(
            graph.adj_src_index._indptr, graph.adj_src_index._sorted_v,
            graph.adj_src_index._sorted_eid, sample_nodes)
    return eids


def graph_saint_random_walk_sample(graph,
                                   hetergraph,
                                   nodes,
                                   max_depth,
                                   alias_name=None,
                                   events_name=None):
    """Implement of graph saint random walk sample.

    First, this function will get random walks path for given nodes and depth.
    Then, it will create subgraph from all sampled nodes.

    Reference Paper: https://arxiv.org/abs/1907.04931

    Args:
        graph: A pgl graph instance
        nodes: Walk starting from nodes
        max_depth: Max walking depth

    Return:
        a subgraph of sampled nodes.
    """
    # the seed of multiprocess for numpy should be reset.
    np.random.seed()
    graph.outdegree()
    # try sample from random nodes
    # nodes=np.random.choice(np.arange(graph.num_nodes, dtype='int64'), size=len(nodes), replace=False)
    nodes = np.random.choice(
        np.arange(
            graph.num_nodes, dtype='int64'), size=20000, replace=False)
    walks = deepwalk_sample(graph, nodes, max_depth, alias_name, events_name)
    sample_nodes = []
    for walk in walks:
        sample_nodes.extend(walk)
    print("length of sample_nodes ", len(sample_nodes))
    sample_nodes = np.unique(sample_nodes)
    print("length of unique sample_nodes ", len(sample_nodes))

    eids = extract_edges_from_nodes(hetergraph, sample_nodes)
    subgraph = hetergraph.subgraph(
        nodes=sample_nodes, eid=eids, with_node_feat=True, with_edge_feat=True)
    #subgraph.node_feat["index"] = np.array(sample_nodes, dtype="int64")
    all_label = graph._node_feat['train_label'][sample_nodes]
    train_index = np.where(all_label > -1)[0]
    train_label = all_label[train_index]
    #print("sample", train_index.shape)
    #print("sample", train_label.shape)
    return subgraph, sample_nodes, train_index, train_label


def graph_saint_hetero(graph, hetergraph, batch_nodes, max_depth=2):
    subgraph, sample_nodes, train_index, train_label = graph_saint_random_walk_sample(
        graph, hetergraph, batch_nodes, max_depth)
    # train_index = subgraph.reindex_from_parrent_nodes(batch_nodes)
    return subgraph, train_index, sample_nodes, train_label


def traverse(item):
    """traverse
    """
    if isinstance(item, list) or isinstance(item, np.ndarray):
        for i in iter(item):
            for j in traverse(i):
                yield j
    else:
        yield item


def flat_node_and_edge(nodes):
    """flat_node_and_edge
    """
    nodes = list(set(traverse(nodes)))
    return nodes


def k_hop_sampler(graph, hetergraph, batch_nodes, samples=[30, 30]):
    # for batch_train_samples, batch_train_labels in batch_info:
    np.random.seed()
    start_nodes = copy.deepcopy(batch_nodes)
    nodes = start_nodes
    edges = []
    for max_deg in samples:
        pred_nodes = graph.sample_predecessor(start_nodes, max_degree=max_deg)

        for dst_node, src_nodes in zip(start_nodes, pred_nodes):
            for src_node in src_nodes:
                edges.append((src_node, dst_node))

        last_nodes = nodes
        nodes = [nodes, pred_nodes]
        nodes = flat_node_and_edge(nodes)
        # Find new nodes
        start_nodes = list(set(nodes) - set(last_nodes))
        if len(start_nodes) == 0:
            break
    nodes = np.unique(np.array(nodes, dtype='int64'))
    eids = extract_edges_from_nodes(hetergraph, nodes)

    subgraph = hetergraph.subgraph(
        nodes=nodes, eid=eids, with_node_feat=True, with_edge_feat=True)
    #sub_node_index = subgraph.reindex_from_parrent_nodes(batch_nodes)
    train_index = subgraph.reindex_from_parrent_nodes(batch_nodes)

    return subgraph, train_index, np.array(nodes, dtype='int64'), None


def dataloader(source_node, label, batch_size=1024):
    index = np.arange(len(source_node))
    np.random.shuffle(index)

    def loader():
        start = 0
        while start < len(source_node):
            end = min(start + batch_size, len(source_node))
            yield source_node[index[start:end]], label[index[start:end]]
            start = end

    return loader


def sample_loader(phase, homograph, hetergraph, gw, source_node, label):
    #print(source_node)
    #print(label)
    if phase == 'train':
        sample_func = graph_saint_hetero
        batch_size = 20000
    else:
        sample_func = k_hop_sampler
        batch_size = 512

    def map_fun(node_label):
        node, label = node_label
        subgraph, train_index, sample_nodes, train_label = sample_func(
            homograph, hetergraph, node)
        #print(train_index.shape)
        #print(sample_nodes.shape)
        #print(sum(subgraph['p2p'].edges[:,0] * subgraph['p2p'].edges[:, 1] == 0) /len(subgraph['p2p'].edges) )
        feed_dict = gw.to_feed(subgraph)
        feed_dict['label'] = label if train_label is None else train_label
        feed_dict['train_index'] = train_index
        feed_dict['sub_node_index'] = sample_nodes
        return feed_dict

    loader = dataloader(source_node, label, batch_size)
    reader = mp_reader_mapper(loader, func=map_fun, num_works=6)

    for feed_dict in reader():
        yield feed_dict


def run_epoch(exe, loss, acc, homograph, hetergraph, gw, train_program,
              test_program, all_label, split_idx, split_real_idx):
    best_acc = 1.0
    for epoch in range(1000):
        for phase in ['train', 'valid', 'test']:
            # if phase == 'train':
            #    continue
            running_loss = []
            running_acc = []
            for feed_dict in sample_loader(
                    phase, homograph, hetergraph, gw,
                    split_real_idx[phase]['paper'],
                    all_label['paper'][split_idx[phase]['paper']]):
                print("train_shape\t", feed_dict['train_index'].shape)
                print("allnode_shape\t", feed_dict['sub_node_index'].shape)
                res = exe.run(
                    train_program if phase == 'train' else test_program,
                    # test_program,
                    feed=feed_dict,
                    fetch_list=[loss.name, acc.name],
                    use_prune=True)
                running_loss.append(res[0])
                running_acc.append(res[1])
                if phase == 'train':
                    print("training_acc %f" % res[1])
            avg_loss = sum(running_loss) / len(running_loss)
            avg_acc = sum(running_acc) / len(running_acc)

            if phase == 'valid':
                if avg_acc > best_acc:
                    fluid.io.save_persistables(exe, './output/checkpoint',
                                               train_program)
                    best_acc = avg_acc
                    print('new best_acc %f' % best_acc)
            print("%d, %s  %f %f" % (epoch, phase, avg_loss, avg_acc))


def main():
    num_class = 349
    num_nodes = 1939743
    start_paper_index = 1203354
    hidden_size = 128

    dataset = PglNodePropPredDataset('ogbn-mag')
    g, all_label = dataset[0]
    homograph = hetero2homo(g)
    for key in g.edge_types_info():
        g[key].outdegree()
        ToShareMemGraph(g[key])

    split_idx = dataset.get_idx_split()
    split_real_idx = copy.deepcopy(split_idx)

    start_paper_index = g.num_node_dict['paper'][1]

    # reindex the original idx of each type of node
    for t, idx in split_real_idx.items():
        for k, v in idx.items():
            split_real_idx[t][k] += g.num_node_dict[k][1]

    homograph._node_feat['train_label'] = -1 * np.ones(
        [homograph.num_nodes, 1], dtype='int64')
    train_label = all_label['paper'][split_idx['train']['paper']]
    train_index = split_real_idx['train']['paper']
    homograph._node_feat['train_label'][train_index] = train_label

    #place = fluid.CUDAPlace(0)
    place = fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()

    additional_paper_feature = g.node_feat_dict[
        'paper'][:, :hidden_size].astype('float32')
    extact_index = (np.arange(start_paper_index, num_nodes)).astype('int32')

    with fluid.program_guard(train_program, startup_program):
        paper_feature = fluid.layers.create_parameter(
            shape=additional_paper_feature.shape,
            dtype='float32',
            default_initializer=fluid.initializer.NumpyArrayInitializer(
                additional_paper_feature),
            name='paper_feature')
        paper_index = fluid.layers.create_parameter(
            shape=extact_index.shape,
            dtype='int32',
            default_initializer=fluid.initializer.NumpyArrayInitializer(
                extact_index),
            name='paper_index')
        #paper_feature.stop_gradient=True
        paper_index.stop_gradient = True

        sub_node_index = fluid.layers.data(
            shape=[-1], dtype='int64', name='sub_node_index')
        train_index = fluid.layers.data(
            shape=[-1], dtype='int64', name='train_index')
        label = fluid.layers.data(shape=[-1], dtype="int64", name='label')
        label = fluid.layers.reshape(label, [-1, 1])
        label.stop_gradient = True
        gw = pgl.heter_graph_wrapper.HeterGraphWrapper(
            name="heter_graph",
            edge_types=g.edge_types_info(),
            node_feat=g.node_feat_info(),
            edge_feat=g.edge_feat_info())
        feat = fluid.layers.create_parameter(
            shape=[num_nodes, hidden_size], dtype='float32')
        # TODO: the paper feature replaced the total feat, not add
        feat = fluid.layers.scatter(
            feat, paper_index, paper_feature, overwrite=False)
        sub_node_feat = fluid.layers.gather(feat, sub_node_index)
        model = RGCNModel(gw, 2, num_class, num_nodes, g.edge_types_info())
        feat = model.forward(sub_node_feat)
        #feat = paper_mask(feat, gw, start_paper_index)
        feat = fluid.layers.gather(feat, train_index)
        loss, logit, acc = softmax_loss(feat, label, num_class)
        opt = fluid.optimizer.AdamOptimizer(learning_rate=0.002)
        opt.minimize(loss)

    test_program = train_program.clone(for_test=True)
    from paddle.fluid.contrib import summary
    summary(train_program)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    # fluid.io.load_persistables(executor=exe, dirname='./output/checkpoint',
    #    main_program=train_program)
    run_epoch(exe, loss, acc, homograph, g, gw, train_program, test_program,
              all_label, split_idx, split_real_idx)

    return None
    feed_dict = gw.to_feed(g)
    #rand_label = (np.random.rand(num_nodes - start_paper_index) >
    #              0.5).astype('int64')
    #feed_dict['label'] = rand_label

    feed_dict['label'] = all_label['paper'][split_idx['train']['paper']]
    feed_dict['train_index'] = split_real_idx['train']['paper']
    #feed_dict['sub_node_index'] = np.arange(num_nodes).astype('int64')
    #feed_dict['paper_index'] = extact_index
    #feed_dict['paper_feature'] = additional_paper_feature

    for epoch in range(10):
        feed_dict['label'] = all_label['paper'][split_idx['train']['paper']]
        feed_dict['train_index'] = split_real_idx['train']['paper']
        for step in range(10):
            res = exe.run(train_program,
                          feed=feed_dict,
                          fetch_list=[loss.name, acc.name])
            print("%d,%d  %f %f" % (epoch, step, res[0], res[1]))
            #print(res[1])

        feed_dict['label'] = all_label['paper'][split_idx['valid']['paper']]
        feed_dict['train_index'] = split_real_idx['valid']['paper']
        res = exe.run(test_program,
                      feed=feed_dict,
                      fetch_list=[loss.name, acc.name])
        print("Test %d,  %f %f" % (epoch, res[0], res[1]))


if __name__ == "__main__":
    main()
