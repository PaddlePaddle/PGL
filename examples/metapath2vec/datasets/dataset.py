# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
sys.path.append("../")
import time
import warnings
import numpy as np
from collections import defaultdict

from pgl.graph_kernel import skip_gram_gen_pair
from pgl.utils.logger import log
from pgl.distributed import DistGraphClient, DistGraphServer
from pgl.utils.data import Dataloader, StreamDataset

from utils.config import prepare_config
from datasets.node import NodeGenerator
from datasets.walk import WalkGenerator
from datasets.pair import PairGenerator

__all__ = ["TrainPairDataset", "InferDataset"]


class TrainPairDataset(StreamDataset):
    def __init__(self, config, ip_list_file, mode="train"):
        self.config = config
        self.ip_list_file = ip_list_file
        self.mode = mode

    def __iter__(self):
        client_id = os.getpid()
        self.graph = DistGraphClient(self.config, self.config.shard_num,
                                     self.ip_list_file, client_id)

        self.generator = PairGenerator(
            self.config,
            self.graph,
            mode=self.mode,
            rank=self._worker_info.fid,
            nrank=self._worker_info.num_workers)

        for data in self.generator():
            yield data


class InferDataset(StreamDataset):
    def __init__(self, config, ip_list_file, mode="infer"):
        self.config = config
        self.ip_list_file = ip_list_file
        self.mode = mode

    def __iter__(self):
        client_id = os.getpid()
        self.graph = DistGraphClient(self.config, self.config.shard_num,
                                     self.ip_list_file, client_id)

        self.generator = NodeGenerator(
            self.config,
            self.graph,
            mode=self.mode,
            rank=self._worker_info.fid,
            nrank=self._worker_info.num_workers)

        for nodes, idx in self.generator():
            nodes = [[nid, nid] for nid in nodes]
            yield nodes


class CollateFn(object):
    def __init__(self):
        pass

    def __call__(self, batch_data):
        src_list = []
        pos_list = []
        for src, pos in batch_data:
            src_list.append(src)
            pos_list.append(pos)

        src_list = np.array(src_list, dtype="int64").reshape(-1, 1)
        pos_list = np.array(pos_list, dtype="int64").reshape(-1, 1)
        return {'src': src_list, 'pos': pos_list}


def test_PairDataset():
    config_file = "../../../config.yaml"
    ip_list_file = "../../../ip_list.txt"
    config = prepare_config(config_file)

    ds = TrainPairDataset(config, ip_list_file)

    loader = Dataloader(
        ds,
        batch_size=4,
        num_workers=1,
        stream_shuffle_size=100,
        collate_fn=CollateFn())
    pairs = []
    start = time.time()
    for batch_data in loader:
        pairs.extend(batch_data)
        print(batch_data)
        time.sleep(10)
    print("total time: %s" % (time.time() - start))
    #  print(pairs)


def test_InferDataset():
    config_file = "../../../config.yaml"
    ip_list_file = "../../../ip_list.txt"
    config = prepare_config(config_file)

    ds = InferDataset(config, ip_list_file)
    loader = Dataloader(ds, batch_size=1, num_workers=1)
    for data in loader:
        print(data[0])
        break


if __name__ == "__main__":
    test_PairDataset()
    #  test_InferDataset()
