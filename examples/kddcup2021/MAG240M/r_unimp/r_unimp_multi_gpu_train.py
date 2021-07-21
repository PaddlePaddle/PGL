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
from dataset.data_generator_r_unimp_sample import MAG240M, DataGenerator

import models
from pgl.utils.logger import log
from utils import save_model, infinite_loop, _create_if_not_exist, load_model
from tensorboardX import SummaryWriter
from collections import defaultdict
import time

    
def train_step(model, loss_fn, batch, dataset):
    graph_list, x, m2v_x, y, label_y, label_idx, = batch

    rd_y = np.random.randint(0, 153, size=label_y.shape)
    rd_m = np.random.rand(label_y.shape[0]) < 0.15
    label_y[rd_m] = rd_y[rd_m]

    x = paddle.to_tensor(x, dtype='float32')
    m2v_x = paddle.to_tensor(m2v_x, dtype='float32')
    y = paddle.to_tensor(y, dtype='int64')
    label_y = paddle.to_tensor(label_y, dtype='int64')
    label_idx = paddle.to_tensor(label_idx, dtype='int64')
    
    graph_list = [(item[0].tensor(), paddle.to_tensor(item[2])) for item in graph_list]
    
    out = model(graph_list, x, m2v_x, label_y, label_idx)
    
    return loss_fn(out, y)

def train(config, do_eval=False):
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    dataset = MAG240M(config)
    evaluator = MAG240MEvaluator()
    
    dataset.prepare_data()
    
    train_iter = DataGenerator(
       dataset=dataset,
       samples=config.samples,
       batch_size=config.batch_size,
       num_workers=config.num_workers,
       data_type="train")

    valid_iter = DataGenerator(
       dataset=dataset,
       samples=config.samples,
       batch_size=config.batch_size,
       num_workers=config.num_workers,
       data_type="eval")

    model_params = dict(config.model.items())
    model_params['m2v_dim'] = config.m2v_dim
    model = getattr(models, config.model.name).GNNModel(**model_params)
    
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
        
    loss_func = F.cross_entropy
    
    opt, lr_scheduler = optim.get_optimizer(parameters=model.parameters(),
                      learning_rate=config.lr,
                      max_steps=config.max_steps,
                      weight_decay=config.weight_decay,
                      warmup_proportion=config.warmup_proportion,
                      clip=config.clip,
                      use_lr_decay=config.use_lr_decay)

    _create_if_not_exist(config.output_path)
    load_model(config.output_path, model)
    swriter = SummaryWriter(os.path.join(config.output_path, 'log'))
    
    if do_eval and paddle.distributed.get_rank() == 0:
        valid_iter = DataGenerator(
                    dataset=dataset,
                    samples=[160] * len(config.samples),
                    batch_size=64,
                    num_workers=config.num_workers,
                    data_type="eval")
        
        r = evaluate(valid_iter, model, loss_func, config, evaluator, dataset)
        log.info(dict(r))
    else:
        best_valid_acc = -1
        for e_id in range(config.epochs):
            loss_temp = []
            for batch in tqdm.tqdm(train_iter.generator()):
                loss = train_step(model, loss_func, batch, dataset)
                
                log.info(loss.numpy()[0])
                loss.backward()
                opt.step()
                opt.clear_gradients()
                loss_temp.append(loss.numpy()[0])
               
            if lr_scheduler is not None:
                    lr_scheduler.step()

            loss = np.mean(loss_temp)
            log.info("Epoch %s Train Loss: %s" % (e_id, loss))
            swriter.add_scalar('loss', loss, e_id)

            if e_id >= config.eval_step and  e_id % config.eval_per_steps == 0 and \
                                            paddle.distributed.get_rank() == 0:
                r = evaluate(valid_iter, model, loss_func, config, evaluator, dataset)
                log.info(dict(r))
                for key, value in r.items():
                    swriter.add_scalar('eval/' + key, value, e_id)
                best_valid_acc = max(best_valid_acc, r['acc'])
                if best_valid_acc == r['acc']:
                    save_model(config.output_path, model, e_id, opt, lr_scheduler)
    swriter.close()
    
    
@paddle.no_grad()
def evaluate(eval_ds, model, loss_fn, config, evaluator, dataset):
    model.eval()
    step = 0
    output_metric = defaultdict(lambda : []) 
    pred_temp = []
    y_temp = []
    
    for batch in eval_ds.generator():
        
        graph_list, x, m2v_x, y, label_y, label_idx, = batch
        x = paddle.to_tensor(x, dtype='float32')
        m2v_x = paddle.to_tensor(m2v_x, dtype='float32')
        y = paddle.to_tensor(y, dtype='int64')
        label_y = paddle.to_tensor(label_y, dtype='int64')
        label_idx = paddle.to_tensor(label_idx, dtype='int64')
        
        graph_list = [(item[0].tensor(), paddle.to_tensor(item[2])) for item in graph_list]
        out = model(graph_list, x, m2v_x, label_y, label_idx)
        loss = loss_fn(out, y)
        
        pred_temp.append(out.numpy())
        y_temp.append(y.numpy())
        output_metric["loss"].append(loss.numpy()[0])

        step += 1
        if step > config.eval_max_steps:
            break

    model.train()
    
    for key, value in output_metric.items():
        output_metric[key] = np.mean(value)
    
    pred_temp = np.concatenate(pred_temp, axis=0)
    y_pred = pred_temp.argmax(axis=-1)
    y_eval = np.concatenate(y_temp, axis=0)
    output_metric['acc'] = evaluator.eval({'y_true': y_eval, 'y_pred': y_pred})['acc']
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
    if args.do_predict:
        predict(config)
    else:
        train(config, args.do_eval)
    
