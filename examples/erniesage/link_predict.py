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
import re
import io

import yaml
import numpy as np
from easydict import EasyDict as edict
from pgl.utils import paddle_helper
from pgl.graph_wrapper import BatchGraphWrapper
import propeller.paddle as propeller
from propeller.paddle.data import Dataset
import paddle.fluid as F
import paddle.fluid.layers as L
import logging
from propeller import log
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.tokenizing_ernie import ErnieTinyTokenizer
log.setLevel(logging.DEBUG)

from dataset.graph_reader import BatchGraphGenerator
from models.encoder import Encoder
from models.pretrain_model_loader import PretrainedModelLoader
from models.loss import Loss
from optimization import optimization


class ERNIESageLinkPredictModel(propeller.train.Model):
    def __init__(self, hparam, mode, run_config):
        self.hparam = hparam
        self.mode = mode
        self.run_config = run_config

    def forward(self, features):
        num_nodes, num_edges, edges, node_feat_index, node_feat_term_ids, user_index, \
            pos_item_index, neg_item_index, user_real_index, pos_item_real_index = features

        node_feat = {"index": node_feat_index, "term_ids": node_feat_term_ids}
        graph_wrapper = BatchGraphWrapper(num_nodes, num_edges, edges,
                                          node_feat)

        encoder = Encoder.factory(self.hparam)
        outputs = encoder([graph_wrapper],
                          [user_index, pos_item_index, neg_item_index])
        user_feat, pos_item_feat, neg_item_feat = outputs

        # loss 
        if self.hparam.neg_type == "batch_neg":
            neg_item_feat = pos_item_feat

        if self.mode is propeller.RunMode.TRAIN:
            return user_feat, pos_item_feat, neg_item_feat
        elif self.mode is propeller.RunMode.PREDICT:
            return user_feat, user_real_index

        elif self.mode is propeller.RunMode.EVAL:
            return user_feat, pos_item_feat, neg_item_feat

    def loss(self, predictions, labels):
        user_feat, pos_item_feat, neg_item_feat = predictions
        loss_func = Loss.factory(self.hparam)
        loss = loss_func(user_feat, pos_item_feat, neg_item_feat)
        return loss

    def backward(self, loss):
        scheduled_lr, _ = optimization(
            loss=loss,
            warmup_steps=int(self.run_config.max_steps *
                             self.hparam['warmup_proportion']),
            num_train_steps=self.run_config.max_steps,
            learning_rate=self.hparam['learning_rate'],
            train_program=F.default_main_program(),
            startup_prog=F.default_startup_program(),
            weight_decay=self.hparam['weight_decay'],
            scheduler="linear_warmup_decay",
            use_fp16=self.hparam.get('use_fp16', 0),
            use_dynamic_loss_scaling=True,
            layer_decay_rate=self.hparam.get("layer_decay_rate", 0.),
            n_layers=self.hparam.ernie_config["num_hidden_layers"])

        propeller.summary.scalar('lr', scheduled_lr)

    def metrics(self, predictions, label):
        return {}


class TrainData(object):
    def __init__(self, graph_work_path):
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        log.info("trainer_id: %s, trainer_count: %s." %
                 (trainer_id, trainer_count))

        edges = np.load(
            os.path.join(graph_work_path, "train_data.npy"), allow_pickle=True)
        # edges is bidirectional.
        train_usr = edges[trainer_id::trainer_count, 0]
        train_ad = edges[trainer_id::trainer_count, 1]
        returns = {"train_data": [train_usr, train_ad]}

        if os.path.exists(os.path.join(graph_work_path, "neg_samples.npy")):
            neg_samples = np.load(
                os.path.join(graph_work_path, "neg_samples.npy"),
                allow_pickle=True)
            if neg_samples.size != 0:
                train_negs = neg_samples[trainer_id::trainer_count]
                returns["train_data"].append(train_negs)
        log.info("Load train_data done.")
        self.data = returns

    def __getitem__(self, index):
        return [data[index] for data in self.data["train_data"]]

    def __len__(self):
        return len(self.data["train_data"][0])


class PredictData(object):
    def __init__(self, num_nodes):
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        train_usr = np.arange(trainer_id, num_nodes, trainer_count)
        #self.data = (train_usr, train_usr)
        self.data = train_usr

    def __getitem__(self, index):
        return [self.data[index], self.data[index]]


def load_tokenizer(ernie_name):
    if "tiny" in config.ernie_name:
        tokenizer = ErnieTinyTokenizer.from_pretrained(ernie_name)
    else:
        tokenizer = ErnieTokenizer.from_pretrained(ernie_name)
    return tokenizer


def train(config):
    # Build Train Data
    data = TrainData(config.graph_work_path)
    train_iter = BatchGraphGenerator(
        graph_wrappers=[1],
        batch_size=config.batch_size,
        data=data,
        samples=config.samples,
        num_workers=config.sample_workers,
        feed_name_list=None,
        use_pyreader=False,
        phase="train",
        graph_data_path=config.graph_work_path,
        shuffle=True,
        neg_type=config.neg_type)
    train_ds = Dataset.from_generator_func(train_iter).repeat(config.epochs)
    dev_ds = Dataset.from_generator_func(train_iter)

    ernie_cfg_dict, ernie_param_path = PretrainedModelLoader.from_pretrained(
        config.ernie_name)

    if "warm_start_from" not in config:
        warm_start_from = ernie_param_path
    else:
        ernie_param_path = config.ernie_param_path

    if "ernie_config" not in config:
        config.ernie_config = ernie_cfg_dict

    ws = propeller.WarmStartSetting(
            predicate_fn=lambda v: os.path.exists(os.path.join(warm_start_from, v.name)),
            from_dir=warm_start_from
        )

    train_ds.name = "train"
    train_ds.data_shapes = [[-1] + list(shape[1:])
                            for shape in train_ds.data_shapes]
    dev_ds.name = "dev"
    dev_ds.data_shapes = [[-1] + list(shape[1:])
                          for shape in dev_ds.data_shapes]

    tokenizer = load_tokenizer(config.ernie_name)
    config.cls_id = tokenizer.cls_id

    propeller.train.train_and_eval(
        model_class_or_model_fn=ERNIESageLinkPredictModel,
        params=config,
        run_config=config,
        train_dataset=train_ds,
        eval_dataset={"eval": dev_ds},
        warm_start_setting=ws, )


def tostr(data_array):
    return " ".join(["%.5lf" % d for d in data_array])


def predict(config):
    # Build Train Data
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))

    num_nodes = int(
        np.load(os.path.join(config.graph_work_path, "num_nodes.npy")))
    data = PredictData(num_nodes)
    predict_iter = BatchGraphGenerator(
        graph_wrappers=[1],
        batch_size=config.infer_batch_size,
        data=data,
        samples=config.samples,
        num_workers=config.sample_workers,
        feed_name_list=None,
        use_pyreader=False,
        phase="predict",
        graph_data_path=config.graph_work_path,
        shuffle=False,
        neg_type=config.neg_type)
    predict_ds = Dataset.from_generator_func(predict_iter)

    predict_ds.name = "predict"
    predict_ds.data_shapes = [[-1] + list(shape[1:])
                              for shape in predict_ds.data_shapes]

    tokenizer = load_tokenizer(config.ernie_name)
    config.cls_id = tokenizer.cls_id

    ernie_cfg_dict, ernie_param_path = PretrainedModelLoader.from_pretrained(
        config.ernie_name)
    config.ernie_config = ernie_cfg_dict

    est = propeller.Learner(ERNIESageLinkPredictModel, config, config)

    id2str = io.open(
        os.path.join(config.graph_work_path, "terms.txt"),
        encoding=config.encoding).readlines()
    fout = io.open(
        "%s/part-%s" % (config.model_dir, trainer_id), "w", encoding="utf8")

    if "infer_model" in config:
        predict_result_iter = est.predict(predict_ds, ckpt_path=config["infer_model"])
    else:
        predict_result_iter = est.predict(predict_ds, ckpt=-1)

    for user_feat, user_real_index in predict_result_iter:
        sri = id2str[int(user_real_index)].strip("\n")
        line = "{}\t{}\n".format(sri, tostr(user_feat))
        fout.write(line)

    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument("--do_predict", action='store_true', default=False)
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    print(config)
    if args.do_predict:
        predict(config)
    else:
        train(config)
