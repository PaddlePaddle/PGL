#-*- coding: utf-8 -*-
import os
import sys
sys.path.append("../")
import time
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
from functools import reduce
from datetime import datetime
from tensorboardX import SummaryWriter

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist

import pgl
from pgl.utils.data import Dataloader
from pgl.utils.logger import log

from ogb.lsc import PCQM4MEvaluator
from ogb.utils import smiles2graph

from utils.config import prepare_config, make_dir
from utils.logger import prepare_logger, log_to_file
import model as M
import dataset as DS

config = prepare_config("./config.yaml", isCreate=False, isSave=False)
env = dist.ParallelEnv()
rank = env.rank
ip_address = config.ip_address.split(',')
os.environ['PADDLE_CURRENT_ENDPOINT'] = ip_address[rank]
os.environ['PADDLE_TRAINER_ENDPOINTS'] = config.ip_address

reg_criterion = paddle.nn.loss.L1Loss()

def data2tensor(batch_dict):
    feed_dict = {}
    for key, value in batch_dict.items():
        if isinstance(value, pgl.Graph):
            feed_dict[key] = value.tensor()
        elif isinstance(value, np.ndarray):
            feed_dict[key] = paddle.to_tensor(value)
        else:
            raise TypeError("can not convert a type of [%s] to paddle Tensor" \
                    % type(value))
    return feed_dict

def train(model, config, loader, optimizer, writer, epoch):
    model.train()
    loss_accum = 0

    for step, (batch_dict, labels, others) in enumerate(loader):
        feed_dict = data2tensor(batch_dict)
        labels = paddle.to_tensor(labels)

        pred = paddle.reshape(model(feed_dict), shape=[-1, ])
        loss = reg_criterion(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        loss_accum += loss.numpy()

        if step % 100 == 0:
            log.info("Epoch: %s | Step: %s | Train loss: %.6f" \
                    % (epoch, step+1, loss_accum / (step+1)) )

    return loss_accum / (step + 1)

@paddle.no_grad()
def evaluate(model, loader, config, mode="valid"):
    model.eval()
    y_true = []
    y_pred = []
    smiles_list = []

    for step, (batch_dict, labels, others) in enumerate(loader):
        feed_dict = data2tensor(batch_dict)

        pred,pretrain_losses = model(feed_dict,return_graph=True)

        y_true.append(labels.reshape(-1, 1))
        y_pred.append(pred.numpy().reshape(-1, 1))

        smiles_list.extend(others['smiles'])

    y_true = np.concatenate(y_true).reshape(-1, )
    y_pred = np.concatenate(y_pred).reshape(-1, )

    input_dict = {"y_true": y_true, "y_pred": y_pred, 'smiles': smiles_list}
    model.train()

    return input_dict

def save_pred_result(output_dir, mode, pred_dict):
    save_file = os.path.join(output_dir, "%s_pred" % mode)
    log.info("saving %s result to npz ..." % mode)
    np.savez_compressed(save_file, pred_dict['y_pred'])

def bn_summary(writer, model, epoch):
    for k, v in model.state_dict().items():
        if "mean" in k or "var" in k:
            v = v.numpy()
            v = np.mean(v)
            writer.add_scalar(k, v, epoch)

def main(config):
    if dist.get_world_size() > 1:
        dist.init_parallel_env()

    if not config.use_cuda:
        paddle.set_device("cpu")

    model = getattr(M, config.model_type)(config)

    if config.warm_start_from:
        log.info("warm start from %s" % config.warm_start_from)
        model.set_state_dict(paddle.load(config.warm_start_from))

    model = paddle.DataParallel(model)

    num_params = sum(p.numel() for p in model.parameters())
    log.info("total Parameters: %s" % num_params)

    if config.lr_mode == "step_decay":
        scheduler = paddle.optimizer.lr.StepDecay(
                learning_rate=config.lr, step_size=config.step_size, gamma=config.gamma)
    elif config.lr_mode == "multistep":
        scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=config.lr, 
                milestones=config.milestones,
                gamma=config.gamma)
    elif config.lr_mode == "piecewise":
        log.info(['boundery: ', config.boundery])
        log.info(['lr_value: ', config.lr_value])
        for i in config.lr_value:
            if not isinstance(i, float):
                raise "lr_value %s is not float number" % i
        scheduler = paddle.optimizer.lr.PiecewiseDecay(config.boundery, config.lr_value)
    elif config.lr_mode == "Reduce":
        scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=config.lr,
                factor=config.factor, patience=config.patience)
    else:
        scheduler = config.lr
    optimizer = getattr(paddle.optimizer, config.optim)(
            learning_rate=scheduler, parameters=model.parameters())

    log.info("loading data")
    ds = getattr(DS, config.dataset_type)(config)

    if config.split_mode == "cross1":
        split_idx = ds.get_cross_idx_split()
        train_ds = DS.Subset(ds, split_idx['cross_train_1'], mode='train')
        valid_ds = DS.Subset(ds, split_idx['cross_valid_1'], mode='valid')
        left_valid_ds = DS.Subset(ds, split_idx['valid_left_1percent'], mode='valid')
        test_ds = DS.Subset(ds, split_idx['test'], mode='test')
    elif config.split_mode == "cross1_few":
        split_idx = ds.get_cross_idx_split()
        train_ds = DS.Subset(ds, split_idx['cross_train_1'][:10000], mode='train')
        valid_ds = DS.Subset(ds, split_idx['cross_train_1'][10000:11000], mode='valid')
        left_valid_ds = DS.Subset(ds, split_idx['cross_train_1'][10000:11000], mode='valid')
        test_ds = DS.Subset(ds, split_idx['cross_train_1'][11000:12000], mode='test')
    elif config.split_mode == "cross2":
        split_idx = ds.get_cross_idx_split()
        train_ds = DS.Subset(ds, split_idx['cross_train_2'], mode='train')
        valid_ds = DS.Subset(ds, split_idx['cross_valid_2'], mode='valid')
        left_valid_ds = DS.Subset(ds, split_idx['valid_left_1percent'], mode='valid')
        test_ds = DS.Subset(ds, split_idx['test'], mode='test')
    else:
        split_idx = ds.get_idx_split()
        train_ds = DS.Subset(ds, split_idx['train'], mode='train')
        valid_ds = DS.Subset(ds, split_idx['valid'], mode='valid')
        left_valid_ds = DS.Subset(ds, split_idx['valid'], mode='valid')
        test_ds = DS.Subset(ds, split_idx['test'], mode='test')

    log.info("Train exapmles: %s" % len(train_ds))
    log.info("Valid exapmles: %s" % len(valid_ds))
    log.info("Test exapmles: %s" % len(test_ds))
    log.info("Left Valid exapmles: %s" % len(left_valid_ds))

    train_loader = Dataloader(train_ds, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, collate_fn=DS.CollateFn(config),
            drop_last=True)

    valid_loader = Dataloader(valid_ds, batch_size=config.valid_batch_size, shuffle=False,
            num_workers=1, collate_fn=DS.CollateFn(config))

    left_valid_loader = Dataloader(left_valid_ds, batch_size=config.valid_batch_size, 
            shuffle=False, num_workers=1, collate_fn=DS.CollateFn(config))

    test_loader = Dataloader(test_ds, batch_size=config.valid_batch_size, shuffle=False,
            num_workers=1, collate_fn=DS.CollateFn(config))

    if config.split_mode is not None:
        valids = {'valid': valid_loader, 'left': left_valid_loader}
    else:
        valids = {'valid': valid_loader}
    
    
    if config.pretrain_tasks:
        pretrain_train_and_eval(model,
            config,
            train_loader,
            valids,
            test_loader,
            optimizer,
            scheduler)
    else:
        train_and_eval(model,
            config,
            train_loader,
            valids,
            test_loader,
            optimizer,
            scheduler)


def pretrain_train_and_eval(model, config, train_loader, valid_loaders, test_loader, optimizer, scheduler):
    evaluator = PCQM4MEvaluator()
    if dist.get_rank() == 0:
        writer = SummaryWriter(config.log_dir)

    best_valid = 1000
    global_step = 0
    header = "%s\n" % config.task_name
    msg_list = []
    epoch_step = len(train_loader)
    topk_best = []
    topk_num = 8
    for i in range(topk_num):
        msg_list.append("")
        topk_best.append([0, 1000])
    # Pretrain
    for epoch in range(1, config.pretrain_epoch + 1):
        model.train()
        if dist.get_rank() == 0:
            bn_summary(writer, model, epoch)
        loss_accum = 0
        loss_dict = {}
        for step, (batch_dict, labels, others) in enumerate(train_loader):
            feed_dict = data2tensor(batch_dict)
            labels = paddle.to_tensor(labels)
            pretrain_losses = model(feed_dict, return_graph=False)
            total_loss = 0
            for name in pretrain_losses:
                if name not in config.pretrain_tasks:
                    continue
                if not name in loss_dict:
                    loss_dict[name] = []
                c_loss = pretrain_losses[name]
                loss_dict[name].append(c_loss.numpy())
                total_loss += c_loss

            total_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            loss_accum += total_loss.numpy()

            if step % 100 == 0:
                log.info("Epoch: %s | Step: %s/%s Pretrain loss: %.6f" \
                        % (epoch, step+1,epoch_step, loss_accum / (step+1)) )
        for name in loss_dict:
            print('pretrain loss', epoch, name, np.mean(loss_dict[name]))
    # Train
    for epoch in range(1, config.epochs + 1):
        model.train()
        alphalist = [config.aux_alpha] * 10 + [config.aux_alpha/2] * 10 +[0]*200
        if dist.get_rank() == 0:
            bn_summary(writer, model, epoch)
        loss_accum = 0
        for step, (batch_dict, labels, others) in enumerate(train_loader):
            feed_dict = data2tensor(batch_dict)
            labels = paddle.to_tensor(labels)
            out, pretrain_losses = model(feed_dict, return_graph=True)
            pred = paddle.reshape(out, shape=[-1, ])
            homo_loss = reg_criterion(pred, labels)
            alpha = alphalist[epoch-1]
            pretrain_loss = alpha * reduce(lambda x,y: x+y,  pretrain_losses.values())
            loss = homo_loss + pretrain_loss
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            loss_accum += loss.numpy()

            if global_step % config.log_step == 0:
                log.info("Epoch: %s | Step: %s/%s | Train loss: %.6f" \
                        % (epoch, step, epoch_step, loss_accum / (step+1)) )
            global_step += 1
        train_mae = loss_accum / (step + 1)
        print("out the training")
        if dist.get_rank() == 0 and config.to_valid_step < epoch:
            valid_dict = evaluate(model, valid_loaders['valid'], config)
            valid_mae = evaluator.eval(valid_dict)["mae"]
            writer.add_scalar('train/mae', train_mae, epoch)
            writer.add_scalar('valid/mae', valid_mae, epoch)

            if config.split_mode is not None:
                left_dict = evaluate(model, valid_loaders['left'], config, 'left_valid')
                left_valid_mae = evaluator.eval(left_dict)["mae"]
                writer.add_scalar('valid/left', left_valid_mae, epoch)

                #  valid_mae = (4.5 * valid_mae + left_valid_mae) / 5.5

            if valid_mae < topk_best[topk_num - 1][1]:
                best_valid = valid_mae

                output_dir = os.path.join(config.output_dir, "%03d" % epoch)
                make_dir(output_dir)
                save_pred_result(output_dir, 'valid', valid_dict)
                save_pred_result(output_dir, 'left_valid', left_dict)

                # if valid is best, save test result
                test_dict = evaluate(model, test_loader, config, mode="test")
                save_pred_result(output_dir, 'test', test_dict)

                save_dir = os.path.join(config.save_dir, "%03d" % epoch)
                make_dir(save_dir)
                ckpt_file = os.path.join(save_dir, "checkpoint.pdparams")
                log.info("saving model checkpoints in %s" % ckpt_file)
                paddle.save(model.state_dict(), ckpt_file)
                #  optim_file = os.path.join(config.save_dir, "optimizer.pdparams")
                #  log.info("saving optimizer checkpoints in %s" % optim_file)
                #  paddle.save(optimizer.state_dict(), optim_file)

                # calculate top n
                for i in range(topk_num):
                    if valid_mae < topk_best[i][1]:
                        topk_best.insert(i, [epoch, valid_mae])
                        k_idx = i
                        break
                to_rm = topk_best[-1]

                tmp_output_dir = os.path.join(config.output_dir, "%03d" % to_rm[0])
                tmp_save_dir = os.path.join(config.save_dir, "%03d" % to_rm[0])
                try:
                    shutil.rmtree(tmp_output_dir)
                    shutil.rmtree(tmp_save_dir)
                except OSError:
                    pass

                topk_best = topk_best[:-1]
                with open(os.path.join(config.output_dir, "ckpt_info"), 'w') as f:
                    for item in topk_best:
                        f.write("%s\n" % item)

                if not config.debug:
                    v_lr = 0.0 if config.lr_mode == "Reduce" else scheduler.get_lr()
                    info = "Epoch: %s | lr: %s | Train: %.6f | Valid: %.6f | Best Valid: %.6f" \
                            % (epoch, v_lr, train_mae, valid_mae, topk_best[0][1])
                    msg_list.insert(k_idx, info)
                    msg_list = msg_list[:-1]
                    to_robot_msg = header + "\n".join(msg_list)
                    os.system("echo '%s' | sh to_robot.sh >/dev/null 2>&1 " % to_robot_msg)

            v_lr = 0.0 if config.lr_mode == "Reduce" else scheduler.get_lr()
            info = "Epoch: %s | lr: %s | Train: %.6f | Valid: %.6f | Best Valid: %.6f" \
                    % (epoch, v_lr, train_mae, valid_mae, topk_best[0][1])
            log.info(info)

            writer.add_scalar('valid/best', topk_best[0][1], epoch)

        if isinstance(scheduler, float):
            pass
        elif config.lr_mode == "Reduce":
            if dist.get_rank() == 0:
                valid_mae = paddle.to_tensor(valid_mae, dtype="float32")
            else:
                valid_mae = paddle.to_tensor(0.0, dtype="float32")
            paddle.distributed.broadcast(valid_mae, 0)
            scheduler.step(valid_mae)
        else:
            scheduler.step()

    
def train_and_eval(model, config, train_loader, valid_loaders, test_loader, optimizer, scheduler):
    evaluator = PCQM4MEvaluator()
    if dist.get_rank() == 0:
        writer = SummaryWriter(config.log_dir)

    best_valid = 1000
    global_step = 0
    header = "%s\n" % config.task_name
    msg_list = []
    epoch_step = len(train_loader)
    topk_best = []
    topk_num = 5
    for i in range(topk_num):
        msg_list.append("")
        topk_best.append([0, 1000])
    for epoch in range(1, config.epochs + 1):
        model.train()
        #  if dist.get_rank() == 0:
            #  bn_summary(writer, model, epoch)
        loss_accum = 0
        train_mae = 1000
        for step, (batch_dict, labels, others) in enumerate(train_loader):
            feed_dict = {}
            for key, value in batch_dict.items():
                if "graph" in key:
                    feed_dict[key] = value.tensor()
                else:
                    feed_dict[key] = paddle.to_tensor(value)
            labels = paddle.to_tensor(labels)

            pred = paddle.reshape(model(feed_dict), shape=[-1, ])
            loss = reg_criterion(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            loss_accum += loss.numpy()

            if global_step % config.log_step == 0:
                log.info("Epoch: %s | Step: %s/%s | Train loss: %.6f" \
                        % (epoch, step, epoch_step, loss_accum / (step+1)) )
            global_step += 1

        train_mae = loss_accum / (step + 1)
        dist.barrier()

        if dist.get_rank() == 0 and config.to_valid_step < epoch:
            valid_dict = evaluate(model, valid_loaders['valid'], config)
            valid_mae = evaluator.eval(valid_dict)["mae"]

            writer.add_scalar('train/mae', train_mae, epoch)
            writer.add_scalar('valid/mae', valid_mae, epoch)

            if config.split_mode is not None:
                left_dict = evaluate(model, valid_loaders['left'], config, 'left_valid')
                left_valid_mae = evaluator.eval(left_dict)["mae"]
                writer.add_scalar('valid/left', left_valid_mae, epoch)

            if valid_mae < topk_best[topk_num - 1][1]:
                best_valid = valid_mae

                output_dir = os.path.join(config.output_dir, "%03d" % epoch)
                make_dir(output_dir)
                save_pred_result(output_dir, 'crossvalid', valid_dict)
                if config.split_mode is not None:
                    save_pred_result(output_dir, 'leftvalid', left_dict)

                # if valid is best, save test result
                test_dict = evaluate(model, test_loader, config, mode="test")
                save_pred_result(output_dir, 'test', test_dict)

                save_dir = os.path.join(config.save_dir, "%03d" % epoch)
                make_dir(save_dir)
                ckpt_file = os.path.join(save_dir, "checkpoint.pdparams")
                log.info("saving model checkpoints in %s" % ckpt_file)
                paddle.save(model.state_dict(), ckpt_file)
                #  optim_file = os.path.join(config.save_dir, "optimizer.pdparams")
                #  log.info("saving optimizer checkpoints in %s" % optim_file)
                #  paddle.save(optimizer.state_dict(), optim_file)

                # calculate top n
                for i in range(topk_num):
                    if valid_mae < topk_best[i][1]:
                        topk_best.insert(i, [epoch, valid_mae])
                        k_idx = i
                        break
                to_rm = topk_best[-1]

                tmp_output_dir = os.path.join(config.output_dir, "%03d" % to_rm[0])
                tmp_save_dir = os.path.join(config.save_dir, "%03d" % to_rm[0])
                try:
                    shutil.rmtree(tmp_output_dir)
                    shutil.rmtree(tmp_save_dir)
                except OSError:
                    pass

                topk_best = topk_best[:-1]
                with open(os.path.join(config.output_dir, "ckpt_info"), 'w') as f:
                    for item in topk_best:
                        f.write("%s\n" % item)

            v_lr = 0.0 if config.lr_mode == "Reduce" else scheduler.get_lr()
            info = "Epoch: %s | lr: %s | Train: %.6f | Valid: %.6f | Best Valid: %.6f" \
                    % (epoch, v_lr, train_mae, valid_mae, topk_best[0][1])
            log.info(info)

            writer.add_scalar('valid/best', topk_best[0][1], epoch)

        if isinstance(scheduler, float):
            pass
        elif config.lr_mode == "Reduce":
            if dist.get_rank() == 0:
                valid_mae = paddle.to_tensor(valid_mae, dtype="float32")
            else:
                valid_mae = paddle.to_tensor(0.0, dtype="float32")
            paddle.distributed.broadcast(valid_mae, 0)
            scheduler.step(valid_mae)
        else:
            scheduler.step()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--task_name", type=str, default="task_name")
    parser.add_argument("--log_id", type=str, default=None)
    args = parser.parse_args()

    if dist.get_rank() == 0:
        config = prepare_config(args.config, isCreate=True, isSave=True)
        if args.log_id is not None:
            config.log_filename = "%s_%s" % (args.log_id, config.log_filename)
        log_to_file(log, config.log_dir, config.log_filename)
    else:
        config = prepare_config(args.config, isCreate=False, isSave=False)

    config.log_id = args.log_id
    main(config)
