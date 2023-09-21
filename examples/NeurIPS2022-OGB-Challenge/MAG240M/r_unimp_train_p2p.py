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
    out_neigh_nodes_mask = paddle.gather(mag_dataset.train_idx_mask,
                                         out_neigh_nodes)
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
        rd_m = paddle.rand(shape=sub_label_y.shape) < config.label_rate  # bool
        sub_label_y[rd_m] = rd_sub_y[rd_m]

    return x, id_x, p2p_x, y, sub_label_y, sub_label_index, pos


def train_step(model, loss_fn, batch, opt):
    graph_list, x, id_x, p2p_x, y, label_y, label_idx, batch_nodes, pos = batch

    out = model(graph_list, x, id_x, p2p_x, label_y, label_idx, batch_nodes,
                pos)
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()
    opt.clear_gradients()
    return loss


def train(config, ensemble_setting=False, do_eval=False):
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    model_params = dict(config.model.items())
    model = getattr(models, config.model.name).GNNModel(**model_params)
    mag_dataset = MAG240M(config, ensemble_setting)
    evaluator = MAG240MEvaluator()

    mag_dataset.prepare_data()
    train_ds = NodeIterDataset(mag_dataset.train_idx)
    eval_ds = NodeIterDataset(mag_dataset.val_idx)

    # Distributed
    train_sampler = DistributedBatchSampler(
        train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
    # train_loader = DataLoader(train_ds, batch_size=config.batch_size,
    #                          shuffle=False, num_workers=0, drop_last=False)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False)

    # model_params = dict(config.model.items())
    # model = getattr(models, config.model.name).GNNModel(**model_params)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    loss_func = F.cross_entropy
    opt, lr_scheduler = \
        optim.get_optimizer(parameters=model.parameters(),
                            learning_rate=config.lr,
                            max_steps=config.max_steps,
                            weight_decay=config.weight_decay,
                            warmup_proportion=config.warmup_proportion,
                            clip=config.clip,
                            use_lr_decay=config.use_lr_decay)

    _create_if_not_exist(config.model_output_path)
    steps = load_model(config.model_output_path, model)
    if do_eval and paddle.distributed.get_rank() == 0:
        eval_loader_new = DataLoader(
            eval_ds,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            drop_last=False)
        eval_ns = NeighborSampler(
            mag_dataset.csc_graphs,
            samples_list=[[160] * 5] * len(config.samples))
        r = evaluate(eval_loader_new, eval_ns, model, loss_func, config,
                     evaluator, mag_dataset)
        log.info(dict(r))
    else:
        samples_list = [[25, 10, 10, 5, 5], [15, 10, 10, 5, 5]]
        ns = NeighborSampler(mag_dataset.csc_graphs, samples_list=samples_list)

        best_valid_acc = -1
        for e_id in range(steps + 1, config.epochs + 1):
            loss_temp = []
            for batch_nodes in train_loader:
                graph_list, neigh_nodes = ns.sample(batch_nodes)
                x, id_x, p2p_x, y, sub_label_y, sub_label_index, pos = get_batch_data(
                    mag_dataset, neigh_nodes, batch_nodes, config)
                batch = (graph_list, x, id_x, p2p_x, y, sub_label_y,
                         sub_label_index, batch_nodes, pos)
                loss = train_step(model, loss_func, batch, opt)
                log.info(float(loss))

                loss_temp.append(float(loss))

            if lr_scheduler is not None:
                lr_scheduler.step()

            loss = np.mean(loss_temp)
            log.info("Epoch %s Train Loss: %s" % (e_id, loss))

            if e_id >= config.eval_step and e_id % config.eval_per_steps == 0:
                r = evaluate(eval_loader, ns, model, loss_func, config,
                             evaluator, mag_dataset)
                # log.info(dict(r))
                # best_valid_acc = max(best_valid_acc, r['acc']) 
                # if best_valid_acc == r['acc']:
                #     save_model(config.model_output_path, model, e_id, opt, lr_scheduler)
                if paddle.distributed.get_rank() == 0:
                    log.info(dict(r))
                    best_valid_acc = max(best_valid_acc, r['acc'])
                    log.info("Best val-acc: %s" % best_valid_acc)
                    if best_valid_acc == r['acc']:
                        save_model(config.model_output_path, model, e_id, opt,
                                   lr_scheduler)
        # save_model(config.model_output_path, model, e_id, opt, lr_scheduler)
        # load_model(config.model_output_path, model)
        # samples = [[160] * 5] * 2
        # eval_loader_new = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
        # eval_ns = NeighborSampler(mag_dataset.csc_graphs, samples_list=samples)
        # r = evaluate(eval_loader_new, eval_ns, model, loss_func, config, evaluator, mag_dataset)
        # if paddle.distributed.get_rank() == 0:
        #     # samples = [[160] * 5] * 2
        #     # eval_loader_new = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
        #     # eval_ns = NeighborSampler(mag_dataset.csc_graphs, samples_list=samples)
        #     # r = evaluate(eval_loader_new, eval_ns, model, loss_func, config, evaluator, mag_dataset)
        #     log.info("Best Result")
        #     log.info(dict(r))


@paddle.no_grad()
def evaluate(eval_loader, neighbor_sampler, model, loss_fn, config, evaluator,
             mag_dataset):
    model.eval()

    output_metric = defaultdict(lambda: [])
    pred_temp = []
    y_temp = []

    for batch_nodes in eval_loader:
        graph_list, neigh_nodes = neighbor_sampler.sample(batch_nodes)
        x, id_x, p2p_x, y, sub_label_y, sub_label_idx, pos  = \
            get_batch_data(mag_dataset, neigh_nodes, batch_nodes, config, eval=True)
        out = model(graph_list, x, id_x, p2p_x, sub_label_y, sub_label_idx,
                    batch_nodes, pos)
        loss = loss_fn(out, y)

        pred_temp.append(out.numpy())
        y_temp.append(y.numpy())
        output_metric["loss"].append(float(loss))

    model.train()

    for key, value in output_metric.items():
        output_metric[key] = np.mean(value)

    pred_temp = np.concatenate(pred_temp, axis=0)
    y_pred = pred_temp.argmax(axis=-1)
    y_eval = np.concatenate(y_temp, axis=0)
    output_metric['acc'] = evaluator.eval({
        'y_true': y_eval,
        'y_pred': y_pred
    })['acc']
    return output_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument(
        "--ensemble_setting", action='store_true', default=False)
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--do_predict", action='store_true', default=False)
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.samples = [int(i) for i in config.samples.split('-')]

    train(config, args.ensemble_setting, args.do_eval)
