# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
import argparse
import time
import os
import math
import glob

import numpy as np
import paddle
from easydict import EasyDict as edict
import pgl
import yaml
from paddle.optimizer import Adam
import tqdm
from pgl.utils.logger import log
from sklearn.metrics import f1_score

from dataset import ShardedDataset


def load(name):
    if name == 'cora':
        dataset = pgl.dataset.CoraDataset()
    elif name == "pubmed":
        dataset = pgl.dataset.CitationDataset("pubmed", symmetry_edges=True)
    elif name == "citeseer":
        dataset = pgl.dataset.CitationDataset("citeseer", symmetry_edges=True)
    elif name == "BlogCatalog":
        dataset = pgl.dataset.BlogCatalogDataset()
    else:
        raise ValueError(name + " dataset doesn't exists")
    dataset.graph.indegree()
    dataset.graph.outdegree()
    dataset.graph = dataset.graph.to_mmap()
    return dataset


class Model(paddle.nn.Layer):
    def __init__(self, num_nodes, embed_size=16, num_classes=39):
        super(Model, self).__init__()

        self.num_nodes = num_nodes

        embed_init = paddle.nn.initializer.Uniform(
            low=-1. / math.sqrt(embed_size), high=1. / math.sqrt(embed_size))
        emb_attr = paddle.ParamAttr(name="node_embedding")
        self.emb = paddle.nn.Embedding(
            num_nodes, embed_size, weight_attr=emb_attr)
        self.linear = paddle.nn.Linear(embed_size, num_classes)

    def forward(self, node_ids):
        node_emb = self.emb(node_ids)
        node_emb.stop_gradient = True
        logits = self.linear(node_emb)
        return logits


def node_classify_generator(graph,
                            all_nodes=None,
                            batch_size=512,
                            epoch=1,
                            shuffle=True):

    if all_nodes is None:
        all_nodes = np.arange(graph.num_nodes)

    def batch_nodes_generator(shuffle=shuffle):
        perm = np.arange(len(all_nodes), dtype=np.int64)
        if shuffle:
            np.random.shuffle(perm)
        start = 0
        while start < len(all_nodes):
            yield all_nodes[perm[start:start + batch_size]]
            start += batch_size

    def wrapper():
        for _ in range(epoch):
            for batch_nodes in batch_nodes_generator():
                # batch_nodes_expanded = np.expand_dims(batch_nodes,
                # -1).astype(np.int64)
                batch_labels = graph.node_feat['group_id'][batch_nodes].astype(
                    np.float32)
                yield [batch_nodes, batch_labels]

    return wrapper


def topk_f1_score(labels,
                  probs,
                  topk_list=None,
                  average="macro",
                  threshold=None):
    assert topk_list is not None or threshold is not None, "one of topklist and threshold should not be None"
    if threshold is not None:
        preds = probs > threshold
    else:
        preds = np.zeros_like(labels, dtype=np.int64)
        for idx, (prob, topk) in enumerate(zip(np.argsort(probs), topk_list)):
            preds[idx][prob[-int(topk):]] = 1
    return f1_score(labels, preds, average=average)


def train(model, data_loader, optim, log_per_step=1000, threshold=0.3):
    model.train()
    total_loss = 0.
    total_sample = 0
    bce_loss = paddle.nn.BCEWithLogitsLoss()
    test_probs_vals, test_labels_vals, test_topk_vals = [], [], []

    for batch, (node, labels) in enumerate(data_loader):
        num_samples = len(node)
        node = paddle.to_tensor(node)
        labels = paddle.to_tensor(labels)
        logits = model(node)
        probs = paddle.nn.functional.sigmoid(logits)
        loss = bce_loss(logits, labels)
        loss.backward()
        optim.step()
        optim.clear_grad()

        topk = labels.sum(-1)
        test_probs_vals.append(probs.numpy())
        test_labels_vals.append(labels.numpy())
        test_topk_vals.append(topk.numpy())

        total_loss += float(loss) * num_samples
        total_sample += num_samples

    test_probs_array = np.concatenate(test_probs_vals)
    test_labels_array = np.concatenate(test_labels_vals)
    test_topk_array = np.concatenate(test_topk_vals)
    test_macro_f1 = topk_f1_score(test_labels_array, test_probs_array,
                                  test_topk_array, "macro", threshold)
    test_micro_f1 = topk_f1_score(test_labels_array, test_probs_array,
                                  test_topk_array, "micro", threshold)
    test_loss_val = total_loss / total_sample
    log.info("Train Loss: %f " % test_loss_val + "Train Macro F1: %f " %
             test_macro_f1 + "Train Micro F1: %f " % test_micro_f1)
    return total_loss / total_sample


@paddle.no_grad()
def test(model, data_loader, log_per_step=1000, threshold=0.3):
    model.eval()
    total_loss = 0.
    total_sample = 0
    bce_loss = paddle.nn.BCEWithLogitsLoss()
    test_probs_vals, test_labels_vals, test_topk_vals = [], [], []

    for batch, (node, labels) in enumerate(data_loader):
        num_samples = len(node)
        node = paddle.to_tensor(node)
        labels = paddle.to_tensor(labels)
        logits = model(node)
        probs = paddle.nn.functional.sigmoid(logits)
        loss = bce_loss(logits, labels)

        topk = labels.sum(-1)
        test_probs_vals.append(probs.numpy())
        test_labels_vals.append(labels.numpy())
        test_topk_vals.append(topk.numpy())

        total_loss += float(loss) * num_samples
        total_sample += num_samples

    test_probs_array = np.concatenate(test_probs_vals)
    test_labels_array = np.concatenate(test_labels_vals)
    test_topk_array = np.concatenate(test_topk_vals)
    test_macro_f1 = topk_f1_score(test_labels_array, test_probs_array,
                                  test_topk_array, "macro", threshold)
    test_micro_f1 = topk_f1_score(test_labels_array, test_probs_array,
                                  test_topk_array, "micro", threshold)
    test_loss_val = total_loss / total_sample
    log.info("\t\tTest Loss: %f " % test_loss_val + "Test Macro F1: %f " %
             test_macro_f1 + "Test Micro F1: %f " % test_micro_f1)
    return test_loss_val, test_macro_f1, test_micro_f1


def load_from_files(model_dir):
    files = glob.glob(
        os.path.join(model_dir, "node_embedding_txt",
                     "node_embedding.block*.txt"))
    emb_table = dict()
    for filename in files:
        for line in open(filename):
            key, value = line.strip(",\n").split("\t")
            key = int(key)
            value = [float(v) for v in value.split(",")]
            emb_table[key] = value

    emb_list = [emb_table[key] for key in range(len(emb_table))]
    emb_arr = np.array(emb_list, dtype=np.float32)
    emb_arr = emb_arr[:, :(emb_arr.shape[1] - 3) // 3]
    return {'emb.weight': emb_arr}


def main(args):
    if not args.use_cuda:
        paddle.set_device("cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    dataset = load(args.dataset)
    graph = dataset.graph

    model = Model(graph.num_nodes, args.embed_size, dataset.num_groups)
    model = paddle.DataParallel(model)

    batch_size = len(dataset.train_index)

    train_steps = int(len(dataset.train_index) / batch_size) * args.epoch
    scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=args.multiclass_learning_rate,
        decay_steps=train_steps,
        end_lr=0.0001)

    optim = Adam(learning_rate=scheduler, parameters=model.parameters())

    if args.load_from_static:
        model.set_state_dict(load_from_files("./model"))
    else:
        model.set_state_dict(paddle.load("model.pdparams"))

    train_data_loader = node_classify_generator(
        graph, dataset.train_index, batch_size=batch_size, epoch=1)
    test_data_loader = node_classify_generator(
        graph, dataset.test_index, batch_size=batch_size, epoch=1)

    best_test_macro_f1 = -1
    for epoch in tqdm.tqdm(range(args.epoch)):
        train_loss = train(model, train_data_loader(), optim)
        test_loss, test_macro_f1, test_micro_f1 = test(model,
                                                       test_data_loader())
        best_test_macro_f1 = max(best_test_macro_f1, test_macro_f1)
    log.info("Best test macro f1 is %s." % best_test_macro_f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepwalk')
    parser.add_argument(
        "--dataset",
        type=str,
        default="BlogCatalog",
        help="dataset (cora, pubmed, BlogCatalog)")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument(
        "--conf",
        type=str,
        default="./config.yaml",
        help="config file for models")
    parser.add_argument("--epoch", type=int, default=1000, help="Epoch")
    parser.add_argument(
        "--load_from_static", action='store_true', help="use_cuda")
    args = parser.parse_args()

    # merge user args and config file 
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(vars(args))
    main(config)
