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
import argparse
import copy
import numpy as np
import pgl
import paddle.fluid as fluid

from paddle.fluid.contrib import summary
from pgl.utils.logger import log
from pgl.utils.share_numpy import ToShareMemGraph
from pgl.contrib.ogb.nodeproppred.dataset_pgl import PglNodePropPredDataset

from rgcn import RGCNModel, cross_entropy_loss
from dataloader import sample_loader


def hetero2homo(heterograph):
    edge = []
    for edge_type in heterograph.edge_types_info():
        log.info('edge_type: {}, shape of edges : {}'.format(
            edge_type, heterograph[edge_type].edges.shape))
        edge.append(heterograph[edge_type].edges)
    edges = np.vstack(edge)
    g = pgl.graph.Graph(num_nodes=heterograph.num_nodes, edges=edges)
    log.info('homo graph nodes %d' % g.num_nodes)
    log.info('homo graph edges %d' % g.num_edges)
    g.outdegree()
    ToShareMemGraph(g)
    return g


def run_epoch(args, exe, fetch_list, homograph, hetergraph, gw, train_program,
              test_program, all_label, split_idx, split_real_idx):
    best_acc = 0.0
    for epoch in range(args.epoch):
        for phase in ['train', 'valid', 'test']:
            running_loss = []
            running_acc = []
            for feed_dict in sample_loader(
                    args, phase, homograph, hetergraph, gw,
                    split_real_idx[phase]['paper'],
                    all_label['paper'][split_idx[phase]['paper']]):
                # print("train_shape\t", feed_dict['train_index'].shape)
                # print("allnode_shape\t", feed_dict['sub_node_index'].shape)
                res = exe.run(train_program
                              if phase == 'train' else test_program,
                              feed=feed_dict,
                              fetch_list=fetch_list,
                              use_prune=True)
                running_loss.append(res[0])
                running_acc.append(res[1])
                if phase == 'train':
                    log.info("training_acc %f" % res[1])
            avg_loss = sum(running_loss) / len(running_loss)
            avg_acc = sum(running_acc) / len(running_acc)

            if phase == 'valid':
                if avg_acc > best_acc:
                    fluid.io.save_persistables(exe, './output/checkpoint',
                                               test_program)
                    best_acc = avg_acc
                    log.info('new best_acc %f' % best_acc)
            log.info("%d, %s  %f %f" % (epoch, phase, avg_loss, avg_acc))


def main(args):
    num_class = 349
    embedding_size = 128
    dataset = PglNodePropPredDataset('ogbn-papers100M')
    g, all_label = dataset[0]
    num_nodes = g.num_nodes

    homograph = hetero2homo(g)
    for key in g.edge_types_info():
        g[key].outdegree()
        ToShareMemGraph(g[key])

    split_idx = dataset.get_idx_split()
    split_real_idx = copy.deepcopy(split_idx)
    # reindex the original idx of each type of node
    for t, idx in split_real_idx.items():
        for k in idx.keys():
            split_real_idx[t][k] += g.num_node_dict[k][1]

    # the num_node_dict record the node nums and the start index of each type of node.
    start_paper_index = g.num_node_dict['paper'][1]
    end_paper_index = start_paper_index + g.num_node_dict['paper'][0]
    # additional feat of paper node, and corresponding index
    additional_paper_feature = g.node_feat_dict[
        'paper'][:, :embedding_size].astype('float32')
    extract_index = (np.arange(start_paper_index,
                               end_paper_index)).astype('int32')

    # extract the train label feat as node feat of homograph
    homograph._node_feat['train_label'] = -1 * np.ones(
        [homograph.num_nodes, 1], dtype='int64')
    # copy train label to homograph node feat
    train_label = all_label['paper'][split_idx['train']['paper']]
    train_index = split_real_idx['train']['paper']
    homograph._node_feat['train_label'][train_index] = train_label

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        gw = pgl.heter_graph_wrapper.HeterGraphWrapper(
            name="heter_graph",
            edge_types=g.edge_types_info(),
            node_feat=g.node_feat_info(),
            edge_feat=g.edge_feat_info())

        # set the paper node feature
        paper_feature = fluid.layers.create_parameter(
            shape=additional_paper_feature.shape,
            dtype='float32',
            default_initializer=fluid.initializer.NumpyArrayInitializer(
                additional_paper_feature),
            name="paper_feature")
        paper_index = fluid.layers.create_parameter(
            shape=extract_index.shape,
            dtype='int32',
            default_initializer=fluid.initializer.NumpyArrayInitializer(
                extract_index),
            name="paper_index")
        paper_feature.stop_gradient = True
        paper_index.stop_gradient = True

        sub_node_index = fluid.layers.data(
            shape=[-1], dtype='int64', name='sub_node_index')
        train_index = fluid.layers.data(
            shape=[-1], dtype='int64', name='train_index')
        label = fluid.layers.data(shape=[-1], dtype="int64", name='label')
        label = fluid.layers.reshape(label, [-1, 1])
        label.stop_gradient = True

        feat = fluid.layers.create_parameter(
            shape=[num_nodes, embedding_size], dtype='float32')
        # NOTE: the paper feature replaced the total feat, not add
        feat = fluid.layers.scatter(
            feat, paper_index, paper_feature, overwrite=False)
        sub_node_feat = fluid.layers.gather(feat, sub_node_index)

        model = RGCNModel(
            graph_wrapper=gw,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_class=num_class,
            edge_types=g.edge_types_info())

        feat = model.forward(sub_node_feat)
        feat = fluid.layers.gather(feat, train_index)
        loss, acc = cross_entropy_loss(feat, label)

        opt = fluid.optimizer.Adam(learning_rate=args.lr)
        opt.minimize(loss)

    test_program = train_program.clone(for_test=True)

    summary(train_program)
    exe = fluid.Executor(place)
    exe.run(startup_program)

    if args.load_pretrain:
        fluid.io.load_persistables(
            executor=exe,
            dirname=os.path.join(args.output_path, 'checkpoint'),
            main_program=test_program)

    fetch_list = [loss.name, acc.name]
    run_epoch(args, exe, fetch_list, homograph, g, gw, train_program,
              test_program, all_label, split_idx, split_real_idx)

    return None


def full_batch(g, gw, all_label, split_idx, split_real_idx, exe, train_program,
               test_program, fetch_list):
    """ The full batch verison of rgcn. No sufficient gpu memory for full batch!
    """
    feed_dict = gw.to_feed(g)
    feed_dict['label'] = all_label['paper'][split_idx['train']['paper']]
    feed_dict['train_index'] = split_real_idx['train']['paper']
    feed_dict['sub_node_index'] = np.arange(g.num_nodes).astype('int64')

    for epoch in range(10):
        feed_dict['label'] = all_label['paper'][split_idx['train']['paper']]
        feed_dict['train_index'] = split_real_idx['train']['paper']

        res = exe.run(train_program, feed=feed_dict, fetch_list=fetch_list)
        log.info("Train %d, %f %f" % (epoch, res[0], res[1]))

        feed_dict['label'] = all_label['paper'][split_idx['valid']['paper']]
        feed_dict['train_index'] = split_real_idx['valid']['paper']
        res = exe.run(test_program, feed=feed_dict, fetch_list=fetch_list)
        log.info("Valid %d,  %f %f" % (epoch, res[0], res[1]))

        feed_dict['label'] = all_label['paper'][split_idx['test']['paper']]
        feed_dict['train_index'] = split_real_idx['test']['paper']
        res = exe.run(test_program, feed=feed_dict, fetch_list=fetch_list)
        log.info("Test %d,  %f %f" % (epoch, res[0], res[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graphsaint with rgcn')
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="output path to save model")
    parser.add_argument(
        "--load_pretrain", action='store_true', help="load pretrained mode")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument("--sample_workers", type=int, default=6)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=512,
        help="sample nums of k-hop of test phase.")
    parser.add_argument(
        "--test_samples",
        type=int,
        nargs='+',
        default=[30, 30],
        help="sample nums of k-hop.")
    args = parser.parse_args()
    log.info(args)
    main(args)
