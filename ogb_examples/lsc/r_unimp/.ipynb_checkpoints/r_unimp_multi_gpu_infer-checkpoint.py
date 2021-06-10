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
import os
import argparse
import traceback
import paddle
import re
import io

import tqdm
import yaml
import pgl
import paddle
import paddle.nn.functional as F 
import numpy as np
import optimization as optim
from ogb.lsc import MAG240MEvaluator

from easydict import EasyDict as edict
from dataset.data_generator_r_unimp_multi_gpu_infer import MAG240M, DataGenerator

import models
from pgl.utils.logger import log
from utils import save_model, infinite_loop, _create_if_not_exist, load_model
from tensorboardX import SummaryWriter
from collections import defaultdict
import time


def infer(config, do_eval=False):
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    dataset = MAG240M(config)
    evaluator = MAG240MEvaluator()
    
    dataset.prepare_data()
    
    model = getattr(models, config.model.name).GNNModel(**dict(config.model.items()))
    
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
        
    loss_func = F.cross_entropy
    
    _create_if_not_exist(config.output_path)
    load_model(config.output_path, model)
    
    
    if paddle.distributed.get_rank() == 0:
        file = 'model_result_temp'
        sudo_label = np.memmap(file, dtype=np.float32, mode='w+',
                              shape=(121751666, 153))
    if do_eval:
        valid_iter = DataGenerator(
                    dataset=dataset,
                    samples=[200] * len(config.samples),
                    batch_size=64,
                    num_workers=config.num_workers,
                    data_type="eval")
        
        r = evaluate(valid_iter, model, loss_func, config, evaluator, dataset)
        log.info("finish eval")
        
        test_iter = DataGenerator(
                    dataset=dataset,
                    samples=[200] * len(config.samples),
                    batch_size=64,
                    num_workers=config.num_workers,
                    data_type="test")
        
        r = evaluate(test_iter, model, loss_func, config, evaluator, dataset)
        log.info("finish test")
    
@paddle.no_grad()
def evaluate(eval_ds, model, loss_fn, config, evaluator, dataset):
    model.eval()
    step = 0
    output_metric = defaultdict(lambda : []) 
    pred_temp = []
    y_temp = []
    file = 'model_result_temp'
    sudo_label = np.memmap(file, dtype=np.float32, mode='r+',
                          shape=(121751666, 153))
    
    for batch in tqdm.tqdm(eval_ds.generator()):
        
        graph_list, x, m2v_x, y, label_y, label_idx, nodes_idx = batch
        x = paddle.to_tensor(x, dtype='float32')
        m2v_x = paddle.to_tensor(m2v_x, dtype='float32')
        y = paddle.to_tensor(y, dtype='int64')
        label_y = paddle.to_tensor(label_y, dtype='int64')
        label_idx = paddle.to_tensor(label_idx, dtype='int64')
        
        graph_list = [(item[0].tensor(), paddle.to_tensor(item[2])) for item in graph_list]
        out = model(graph_list, x, m2v_x, label_y, label_idx)
        out = F.softmax(out)
        sudo_label[nodes_idx] = out.numpy()

    sudo_label.flush()
    model.train()
    return output_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--do_predict", action='store_true', default=False)
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.samples = [int(i) for i in config.samples.split('-')]

    print(config)
    infer(config, args.do_eval)

