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

import yaml
import numpy as np
from easydict import EasyDict as edict
from pgl.utils.logger import log
from pgl.utils import paddle_helper

from learner import Learner
from models.model_factory import Model
from dataset.graph_reader import GraphGenerator 


class TrainData(object):
    def __init__(self, graph_path):
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        log.info("trainer_id: %s, trainer_count: %s." % (trainer_id, trainer_count))

        edges = np.load(os.path.join(graph_path, "edges.npy"), allow_pickle=True)
        # edges is bidirectional.
        train_usr = edges[trainer_id::trainer_count, 0]
        train_ad = edges[trainer_id::trainer_count, 1]
        returns = {
            "train_data": [train_usr, train_ad]
        }

        if os.path.exists(os.path.join(graph_path, "neg_samples.npy")):
            neg_samples = np.load(os.path.join(graph_path, "neg_samples.npy"), allow_pickle=True)
            if neg_samples.size != 0:
                train_negs = neg_samples[trainer_id::trainer_count]
                returns["train_data"].append(train_negs)
        log.info("Load train_data done.")
        self.data = returns

    def __getitem__(self, index):
        return [ data[index] for data in self.data["train_data"]]

    def __len__(self):
        return len(self.data["train_data"][0])


def main(config):
    # Select Model
    model = Model.factory(config)

    # Build Train Edges
    data = TrainData(config.graph_path)

    # Build Train Data
    train_iter = GraphGenerator(
        graph_wrappers=model.graph_wrappers,
        batch_size=config.batch_size,
        data=data,
        samples=config.samples,
        num_workers=config.sample_workers,
        feed_name_list=[var.name for var in model.feed_list],
        use_pyreader=config.use_pyreader,
        phase="train",
        graph_data_path=config.graph_path,
        shuffle=True)

    log.info("build graph reader done.")

    learner = Learner.factory(config.learner_type)
    learner.build(model, train_iter, config)

    learner.start()
    learner.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    print(config)
    main(config)
