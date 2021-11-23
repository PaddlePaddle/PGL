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
import os
import sys
import time
import tqdm
import yaml
import argparse
import numpy as np

import pgl
from pgl.utils.logger import log
from pgl.utils.data import Dataloader, StreamDataset

import paddle
import paddle.nn as nn
from paddle.optimizer import Adam

from utils.config import prepare_config
from utils.logger import log_to_file
import models as M
import datasets.dataset as DS


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
def inference(model, loader, save_dir):
    model.eval()
    save_file = os.path.join(save_dir, "embedding.txt")
    cc = 0
    with open(save_file, "w") as writer:
        for step, feed_dict in enumerate(loader()):
            feed_dict = data2tensor(feed_dict)
            pred = model(feed_dict)
            node_ids, node_embed = model.get_embedding()
            for nid, vec in zip(node_ids.numpy().reshape(-1),
                                node_embed.numpy()):
                str_vec = ' '.join(map(str, vec))
                writer.write("%s\t%s\n" % (nid, str_vec))
                cc += 1

                if cc % 10000 == 0:
                    log.info("%s nodes have been processed" % (cc))

    log.info("total %s nodes have been processed" % cc)
    log.info("node representations are saved in %s" % save_file)
    model.train()


def main(config, ip_list_file, save_dir, infer_from):
    ds = getattr(DS, config.dataset_type)(config, ip_list_file, mode="infer")
    loader = Dataloader(
        ds,
        batch_size=config.batch_pair_size,
        num_workers=config.num_workers,
        collate_fn=getattr(DS, config.collatefn)(config, mode="gpu"))

    log.info("building model")
    config.embed_type = "BaseEmbedding"
    log.info("embed type is %s" % config.embed_type)
    model = getattr(M, config.model_type)(config, mode="gpu")

    if os.path.exists(infer_from):
        log.info("infer from %s" % infer_from)
        model.set_state_dict(paddle.load(infer_from))
    else:
        raise ValueError("infer path %s is not existed!" % infer_from)

    inference(model, loader, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GraphRec')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--ip", type=str, default="./ip_list.txt")
    parser.add_argument("--task_name", type=str, default="graph_rec")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--infer_from", type=str, default=None)
    args = parser.parse_args()

    config = prepare_config(args.config)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(config, args.ip, args.save_dir, args.infer_from)
