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
import model# import LabelGraphGCN
from pgl import data_loader
from pgl.utils.logger import log
import paddle.fluid as fluid
import numpy as np
import time
import argparse
from build_model import build_model
import yaml
from easydict import EasyDict as edict
import tqdm

def normalize(feat):
    return feat / np.maximum(np.sum(feat, -1, keepdims=True), 1)


def load(name, normalized_feature=True):
    if name == 'cora':
        dataset = data_loader.CoraDataset()
    elif name == "pubmed":
        dataset = data_loader.CitationDataset("pubmed", symmetry_edges=True)
    elif name == "citeseer":
        dataset = data_loader.CitationDataset("citeseer", symmetry_edges=True)
    else:
        raise ValueError(name + " dataset doesn't exists")

    indegree = dataset.graph.indegree()
    norm = np.maximum(indegree.astype("float32"), 1)
    norm = np.power(norm, -0.5)
    dataset.graph.node_feat["norm"] = np.expand_dims(norm, -1)
    dataset.graph.node_feat["words"] =  normalize(dataset.graph.node_feat["words"])
    return dataset


def main(args, config):
    dataset = load(args.dataset, args.feature_pre_normalize)
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()
    with fluid.program_guard(train_program, startup_program):
        with fluid.unique_name.guard():
            gw, loss, acc = build_model(dataset,
                                config=config,
                                phase="train",
                                main_prog=train_program)

    test_program = fluid.Program()
    with fluid.program_guard(test_program, startup_program):
        with fluid.unique_name.guard():
            _gw, v_loss, v_acc = build_model(dataset,
                config=config,
                phase="test",
                main_prog=test_program)

    test_program = test_program.clone(for_test=True)

    exe = fluid.Executor(place)


    train_index = dataset.train_index
    train_label = np.expand_dims(dataset.y[train_index], -1)
    train_index = np.expand_dims(train_index, -1)

    val_index = dataset.val_index
    val_label = np.expand_dims(dataset.y[val_index], -1)
    val_index = np.expand_dims(val_index, -1)

    test_index = dataset.test_index
    test_label = np.expand_dims(dataset.y[test_index], -1)
    test_index = np.expand_dims(test_index, -1)

    dur = []

    # Feed data
    feed_dict = gw.to_feed(dataset.graph)


    best_test = []
 
    for run in range(args.runs):
        exe.run(startup_program)
        cal_val_acc = []
        cal_test_acc = []
        cal_val_loss = []
        cal_test_loss = []
        for epoch in tqdm.tqdm(range(args.epoch)):
            feed_dict["node_index"] = np.array(train_index, dtype="int64")
            feed_dict["node_label"] = np.array(train_label, dtype="int64")
            train_loss, train_acc = exe.run(train_program,
                                        feed=feed_dict,
                                        fetch_list=[loss, acc],
                                        return_numpy=True)


            feed_dict["node_index"] = np.array(val_index, dtype="int64")
            feed_dict["node_label"] = np.array(val_label, dtype="int64")
            val_loss, val_acc = exe.run(test_program,
                                    feed=feed_dict,
                                    fetch_list=[v_loss, v_acc],
                                    return_numpy=True)

            cal_val_acc.append(val_acc[0])
            cal_val_loss.append(val_loss[0])

            feed_dict["node_index"] = np.array(test_index, dtype="int64")
            feed_dict["node_label"] = np.array(test_label, dtype="int64")
            test_loss, test_acc = exe.run(test_program,
                                  feed=feed_dict,
                                  fetch_list=[v_loss, v_acc],
                                  return_numpy=True)

            cal_test_acc.append(test_acc[0])
            cal_test_loss.append(test_loss[0])

     
        log.info("Runs %s: Model: %s Best Test Accuracy: %f" % (run, config.model_name,
                  cal_test_acc[np.argmin(cal_val_loss)]))

        best_test.append(cal_test_acc[np.argmin(cal_val_loss)])
    log.info("Dataset: %s Best Test Accuracy: %f ( stddev: %f )" % (args.dataset, np.mean(best_test), np.std(best_test)))
    print("Dataset: %s Best Test Accuracy: %f ( stddev: %f )" % (args.dataset, np.mean(best_test), np.std(best_test)))
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarking Citation Network')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="dataset (cora, pubmed)")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument("--conf", type=str, help="config file for models")
    parser.add_argument("--epoch", type=int, default=200, help="Epoch")
    parser.add_argument("--runs", type=int, default=10, help="runs")
    parser.add_argument("--feature_pre_normalize", type=bool, default=True, help="pre_normalize feature")
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    log.info(args)
    main(args, config)
