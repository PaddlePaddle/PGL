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

import argparse
import os
import yaml
import paddle
from paddle.io import DataLoader, DistributedBatchSampler
import paddle.nn.functional as F
import numpy as np
import optimization as optim
from ogb.lsc import MAG240MEvaluator
from easydict import EasyDict as edict
from pgl.utils.logger import log
from collections import defaultdict

import models
from dataset.new_dataset_p2p import MAG240M, NodeIterDataset, NeighborSampler
from utils import _create_if_not_exist, load_model, save_model


def get_batch_data(mag_dataset, neigh_nodes, batch_nodes, config, eval=False):
    out_neigh_nodes = neigh_nodes[batch_nodes.shape[0]:]
    out_neigh_nodes_mask = paddle.gather(mag_dataset.train_idx_mask, out_neigh_nodes)
    # sub_label_index = paddle.where(out_neigh_nodes_mask)[0].reshape([-1])
    sub_label_index = paddle.where(out_neigh_nodes_mask == 1)[0].reshape([-1])
    sub_label_y = mag_dataset.y[out_neigh_nodes[sub_label_index]]
    sub_label_index = sub_label_index + len(batch_nodes)

    x = mag_dataset.x[neigh_nodes]
    id_x = mag_dataset.id_x[neigh_nodes]
    p2p_x = mag_dataset.p2p_x[neigh_nodes]
    # 以下行可注释
    # x = paddle.to_tensor(x, dtype="float16")
    x = paddle.cast(x, dtype="float32")
    # 注意合入最新的MAG240M from Zhengjie
    # id_x = paddle.to_tensor(id_x, dtype="float16")
    id_x = paddle.cast(id_x, dtype="float32")
    p2p_x = paddle.cast(p2p_x, dtype="float32")
    y = mag_dataset.y[batch_nodes]

    pos = 2021 - mag_dataset.year[neigh_nodes]
    pos = mag_dataset.pos[pos]
    
    if not eval:
        rd_sub_y = paddle.randint_like(sub_label_y, 0, 153)
        rd_m = paddle.rand(shape=sub_label_y.shape) < config.label_rate # bool
        sub_label_y[rd_m] = rd_sub_y[rd_m]

    return x, id_x, p2p_x, y, sub_label_y, sub_label_index, pos

def infer(config,  do_eval=False):
    _create_if_not_exist(config.model_result_path)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    model_params = dict(config.model.items())
    model = getattr(models, config.model.name).GNNModel(**model_params)
    mag_dataset = MAG240M(config, ensemble_setting=True)
    mag_dataset.prepare_data()

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if paddle.distributed.get_rank() == 0:
        file = 'model_result_temp'
        if not os.path.exists(file):
            sudo_label = np.memmap(file, dtype=np.float32, mode='w+',
                                shape=(121751666, 153))

    load_model(config.model_output_path, model)
    # single
    if do_eval and paddle.distributed.get_rank() == 0:
        # cv valid idx
        # valid_name = os.path.join(config.valid_path, config.valid_name)
        # val_idx = np.load(valid_name)
        eval_ds = NodeIterDataset(mag_dataset.val_idx)
        test_ds = NodeIterDataset(mag_dataset.test_idx)
        test_dev_ds = NodeIterDataset(mag_dataset.test_dev_idx)
        eval_loader_new = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
        test_loader_new = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
        test_dev_loader_new = DataLoader(test_dev_ds, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
        ns = NeighborSampler(mag_dataset.csc_graphs, samples_list=[[160] * 5] * len(config.samples))
        r = evaluate(eval_loader_new, ns, model, mag_dataset)
        log.info("finish eval")
        r = evaluate(test_loader_new, ns, model, mag_dataset)
        log.info("finish test-challenge")
        r = evaluate(test_dev_loader_new, ns, model, mag_dataset)
        log.info("finish test-dev")

@paddle.no_grad()
def evaluate(eval_loader, neighbor_sampler, model, mag_dataset):
    model.eval()
    output_metric = defaultdict(lambda: [])
    pred_temp = []
    y_temp = []
    file = 'model_result_temp'
    sudo_label = np.memmap(file, dtype=np.float32, mode='r+',
                          shape=(121751666, 153))
    for batch_nodes in eval_loader:
        graph_list, neigh_nodes = neighbor_sampler.sample(batch_nodes)
        x, id_x, p2p_x, y, sub_label_y, sub_label_idx, pos  = \
            get_batch_data(mag_dataset, neigh_nodes, batch_nodes, config, eval=True)
        out = model(graph_list, x, id_x, p2p_x,  sub_label_y, sub_label_idx, batch_nodes, pos)
        out = F.softmax(out)
        sudo_label[batch_nodes.numpy().tolist()] = out.numpy()
    sudo_label.flush()
    return output_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument("--ensemble_setting", action='store_true', default=False)
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--do_predict", action='store_true', default=False)
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.samples = [int(i) for i in config.samples.split('-')]

    infer(config, args.do_eval)
