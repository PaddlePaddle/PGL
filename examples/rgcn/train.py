# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

import pgl
from pgl import data_loader
from pgl.utils.logger import log
import paddle.fluid as fluid
import numpy as np
import time
import random
import argparse
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from Dataset import Dataset
from layer import RGCNConv

def model(args, dataset):
    """
    doc
    """
    node_index = fluid.layers.data(
        "node_index",
        shape=[None, 1],
        dtype="int64",
        append_batch_size=False)
    node_label = fluid.layers.data(
        "node_label",
        shape=[None, 1],
        dtype="int64",
        append_batch_size=False)

    gw = pgl.heter_graph_wrapper.HeterGraphWrapper(
        name="heter_graph",
        place=args.place,
        edge_types=dataset.graph.edge_types_info(),
        node_feat=dataset.graph.node_feat_info(),
        edge_feat=dataset.graph.edge_feat_info())


    output = RGCNConv(gw, args.hidden_size, 
                        args.hidden_size, 
                        etypes=dataset.graph.edge_types_info(), 
                        num_nodes=dataset.graph.num_nodes,
                        num_bases=args.num_bases, )
    output = RGCNConv(gw, args.hidden_size, 
                        args.hidden_size,
                        etypes=dataset.graph.edge_types_info(), 
                        num_nodes=dataset.graph.num_nodes,
                        num_bases=args.num_bases, )

    linear = fluid.layers.create_parameter(
            shape=[args.hidden_size, args.num_classes],
            dtype='float32',
            name='linear') 

    output = fluid.layers.matmul(output, linear)

    pred = fluid.layers.gather(output, node_index) 

    loss, _ = fluid.layers.softmax_with_cross_entropy(
        logits=pred, label=node_label, return_softmax=True)

    pred = fluid.layers.softmax(pred, axis=- 1)
    acc = fluid.layers.accuracy(input=pred, label=node_label, k=1)
    loss = fluid.layers.mean(loss)

    feed_dict = gw.to_feed(dataset.graph)
    fetch_list = loss, acc, pred

    return feed_dict, fetch_list


def test(args, test_program, exe, dataset, feed_dict, fetch_list, save_predict_file=False):
    """
    Testing.
    """
    test_index = dataset.test_index
    test_label = dataset.node_labels[test_index]
    
    test_index = np.expand_dims(test_index, -1)

    feed_dict["node_index"] = np.array(test_index, dtype="int64")
    feed_dict["node_label"] = np.array(test_label, dtype="int64")
    test_loss, test_acc, test_pred = exe.run(test_program,
                                feed=feed_dict,
                                fetch_list=fetch_list,
                                return_numpy=True)

    y_label = np.squeeze(test_label, axis=-1)

    y_pred = np.zeros(len(test_pred), dtype=np.int32)
    for i in range(len(test_pred)):
        y_pred[i] = np.argmax(test_pred[i])

    p_mi = precision_score(y_label, y_pred, average='micro')
    p_ma = precision_score(y_label, y_pred, average='macro')
    r_mi = recall_score(y_label, y_pred, average='micro')
    r_ma = recall_score(y_label, y_pred, average='macro')
    f_mi = f1_score(y_label, y_pred, average='micro')
    f_ma = f1_score(y_label, y_pred, average='macro')
    
    if save_predict_file:
        with open(args.predict_output, 'w') as f:
            f.write('\t'.join(map(str, ["test_index", "test_label", "test_pred"])) + '\n')

        for test_i, test_label, test_pred in zip(test_index, y_label, y_pred):
            with open(args.predict_output, 'a') as f:
                f.write('\t'.join(map(str, [test_i[0], test_label, test_pred])) + '\n')

    log.info("Test Accuracy: %f " % (test_acc))
    log.info("Micro: Precision: %f Recall: %f F1: %f " % (p_mi, r_mi, f_mi))
    log.info("Macro: Precision: %f Recall: %f F1: %f \n" % (p_ma, r_ma, f_ma))


def main(args):
    """
    doc
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    fluid.default_startup_program().random_seed = args.seed
    fluid.default_main_program().random_seed = args.seed
    fluid.Program().random_seed = args.seed
    dataset = Dataset(args)

    args.place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    log.info("Place: %s " % (args.place))

    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        feed_dict, [loss, acc, pred] = model(args, dataset)

    test_program = train_program.clone(for_test=True)
    with fluid.program_guard(train_program, startup_program):
        adam = fluid.optimizer.Adam(
            learning_rate=args.lr,
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.001)
                )
        adam.minimize(loss)

    exe = fluid.Executor(args.place)
    exe.run(startup_program)

    train_index = dataset.train_index
    train_label = dataset.node_labels[train_index]
    train_index = np.expand_dims(train_index, -1)

    val_index = dataset.val_index
    val_label = dataset.node_labels[val_index]
    val_index = np.expand_dims(val_index, -1)

    dur = []
    for epoch in range(args.epochs):
        t0 = time.time()
        feed_dict["node_index"] = np.array(train_index, dtype="int64")
        feed_dict["node_label"] = np.array(train_label, dtype="int64")
        train_loss, train_acc = exe.run(train_program,
                                        feed=feed_dict,
                                        fetch_list=[loss, acc],
                                        return_numpy=True)
        
        time_per_epoch = 1.0 * (time.time() - t0)
        dur.append(time_per_epoch)

        if epoch % args.eval_epochs == 0:
            feed_dict["node_index"] = np.array(val_index, dtype="int64")
            feed_dict["node_label"] = np.array(val_label, dtype="int64")
            val_loss, val_acc, val_pred = exe.run(test_program,
                                        feed=feed_dict,
                                        fetch_list=[loss, acc, pred],
                                        return_numpy=True)

            log.info("Epoch %d " % epoch + "(%.5lf sec) " % np.mean(dur) +
                    "Train Loss: %f " % train_loss + "Train Acc: %f " % train_acc
                    + "Val Loss: %f " % val_loss + "Val Acc: %f " % val_acc)
        
        if epoch % args.test_epochs == 0:
            test(args, test_program, exe, dataset, feed_dict, [loss, acc, pred])

    test(args, test_program, exe, dataset, feed_dict, [loss, acc, pred], save_predict_file=True)

    feed_list = list(feed_dict.keys())
    feed_list.remove("node_label")
    for e in dataset.graph.edge_types_info():
        var = "heter_graph/" + e
        feed_list.remove(var + "/edges_dst")
        feed_list.remove(var + "/indegree")
        feed_list.remove(var + "/graph_lod")
        feed_list.remove(var + "/num_graph")

    fluid.io.save_inference_model(
            args.model_output, feed_list, [pred], exe,
            params_filename="params",
            model_filename="model",
            main_program=test_program)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--use_cuda", action='store_true', default=True, help="use_cuda")
    parser.add_argument("--dataset", type=str, default="./data", help="dataset dir")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch")
    parser.add_argument("--eval_epochs", type=int, default=1, help="Epoch")
    parser.add_argument("--test_epochs", type=int, default=200, help="Epoch")
    parser.add_argument("--model_output", type=str, default="./best_model/", help="Path to save checkpoints.")
    parser.add_argument("--predict_output", type=str, default="./test.result.tsv", help="")
    parser.add_argument("--lr", type=float, default=0.01, help="pre_normalize feature")
    parser.add_argument("--hidden_size", type=int, default=64, help="pre_normalize feature")
    parser.add_argument("--num_bases", type=int, default=0, help="")
    parser.add_argument("--num_classes", type=int, default=2, help="")
    parser.add_argument("--seed", type=int, default=0)
    
    
    args = parser.parse_args()
    log.info(args)
    main(args)
