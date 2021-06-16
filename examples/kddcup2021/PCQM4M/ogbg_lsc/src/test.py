#-*- coding: utf-8 -*-
import os
import sys
sys.path.append("../")
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist

import pgl
from pgl.utils.data import Dataloader
from pgl.utils.logger import log

from ogb.lsc import PCQM4MDataset, PCQM4MEvaluator
from ogb.utils import smiles2graph

from utils.config import prepare_config, make_dir
from utils.logger import prepare_logger, log_to_file
import model as M
import dataset as DS

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

@paddle.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    for step, (batch_dict, labels, others) in enumerate(loader):
        feed_dict = data2tensor(batch_dict)

        pred = model(feed_dict)

        y_true.append(labels.reshape(-1, 1))
        y_pred.append(pred.numpy().reshape(-1, 1))

    y_true = np.concatenate(y_true).reshape(-1, )
    y_pred = np.concatenate(y_pred).reshape(-1, )

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return input_dict

@paddle.no_grad()
def infer(config):
    model = getattr(M, config.model_type)(config)

    log.info("infer model from %s" % config.infer_from)
    model.set_state_dict(paddle.load(config.infer_from))

    log.info("loading data")
    ds = getattr(DS, config.dataset_type)(config)

    split_idx = ds.get_idx_split()
    train_ds = DS.Subset(ds, split_idx['train'], mode='train')
    valid_ds = DS.Subset(ds, split_idx['valid'], mode='valid')
    test_ds = DS.Subset(ds, split_idx['test'], mode='test')

    log.info("Train exapmles: %s" % len(train_ds))
    log.info("Valid exapmles: %s" % len(valid_ds))
    log.info("Test exapmles: %s" % len(test_ds))

    train_loader = Dataloader(train_ds, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, collate_fn=DS.CollateFn(config),
            drop_last=True)

    valid_loader = Dataloader(valid_ds, batch_size=config.valid_batch_size, shuffle=False,
            num_workers=1, collate_fn=DS.CollateFn(config))

    test_loader = Dataloader(test_ds, batch_size=config.valid_batch_size, shuffle=False,
            num_workers=1, collate_fn=DS.CollateFn(config))

    try:
        task_name = config.infer_from.split("/")[-2]
    except:
        task_name = "ogb_kdd"
    log.info("task_name: %s" % task_name)

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    # ---------------- valid ----------------------- #
    #  log.info("validating ...")
    #  pred_dict = evaluate(model, valid_loader)
    #
    #  log.info("valid MAE: %s" % evaluator.eval(pred_dict)["mae"])
    #  valid_output_path = os.path.join(config.output_dir, task_name)
    #  make_dir(valid_output_path)
    #  valid_output_file = os.path.join(valid_output_path, "valid_mae.txt")
    #
    #  log.info("saving valid result to %s" % valid_output_file)
    #  with open(valid_output_file, 'w') as f:
    #      for y_pred, idx in zip(pred_dict['y_pred'], split_idx['valid']):
    #          smiles, label = ds.raw_dataset[idx]
    #          f.write("%s\t%s\t%s\n" % (y_pred, label, smiles))
    #
    # ---------------- test ----------------------- #

    log.info("testing ...")
    pred_dict = evaluate(model, test_loader)

    test_output_path = os.path.join(config.output_dir, task_name)
    make_dir(test_output_path)
    test_output_file = os.path.join(test_output_path, "test_mae.txt")

    log.info("saving test result to %s" % test_output_file)
    with open(test_output_file, 'w') as f:
        for y_pred, idx in zip(pred_dict['y_pred'], split_idx['test']):
            smiles, label = ds.raw_dataset[idx]
            f.write("%s\t%s\n" % (y_pred, smiles))

    log.info("saving submition format to %s" % test_output_path)
    evaluator.save_test_submission({'y_pred': pred_dict['y_pred']}, test_output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--task_name", type=str, default="task_name")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--log_id", type=str, default=None)
    args = parser.parse_args()

    config = prepare_config(args.config, isCreate=False, isSave=False)
    infer(config)

