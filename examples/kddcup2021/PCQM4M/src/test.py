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
def infer(config, output_path):
    model = getattr(M, config.model_type)(config)

    log.info("infer model from %s" % config.infer_from)
    model.set_state_dict(paddle.load(config.infer_from))

    log.info("loading data")
    ds = getattr(DS, config.dataset_type)(config)

    split_idx = ds.get_idx_split()
    test_ds = DS.Subset(ds, split_idx['test'], mode='test')
    log.info("Test exapmles: %s" % len(test_ds))

    test_loader = Dataloader(test_ds, batch_size=config.valid_batch_size, shuffle=False,
            num_workers=1, collate_fn=DS.CollateFn(config))

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    # ---------------- test ----------------------- #
    log.info("testing ...")
    pred_dict = evaluate(model, test_loader)

    test_output_path = os.path.join(config.output_dir, config.task_name)
    make_dir(test_output_path)
    test_output_file = os.path.join(test_output_path, "test_pred.npz")

    log.info("saving test result to %s" % test_output_file)
    np.savez_compressed(test_output_file, pred_dict['y_pred'].astype(np.float32))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--task_name", type=str, default="task_name")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="./")
    args = parser.parse_args()

    config = prepare_config(args.config, isCreate=False, isSave=False)
    make_dir(args.output_path)
    infer(config, args.output_path)

