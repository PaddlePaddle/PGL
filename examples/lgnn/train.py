# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.utils.data import Dataloader

from model import LGNN
from cora_binary import CoraBinary


def main():
    train_set = CoraBinary()
    training_loader = Dataloader(train_set, batch_size=1)

    model = LGNN(radius=3)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=4e-3)
    for i in range(20):
        all_loss = []
        all_acc = []
        #for idx, (g, lg, label) in enumerate(training_loader):
        for idx, inputs in enumerate(training_loader):
            #print(xxx)
            (p_g, p_lg, label) = inputs[0]
            # Generate the line graph.
            p_g.tensor()
            p_lg.tensor()
            # Create paddle tensors
            label = paddle.to_tensor(label)
            # Forward
            z = model(p_g, p_lg)

            # Calculate loss:
            # Since there are only two communities, there are only two permutations
            #  of the community labels.
            loss_perm1 = F.cross_entropy(z, label)
            loss_perm2 = F.cross_entropy(z, 1 - label)
            loss = paddle.minimum(loss_perm1, loss_perm2)

            # Calculate accuracy:
            # pred = paddle.max(z, 1)
            pred = paddle.where(z[:, 0] > z[:, 1],
                                paddle.zeros_like(z[:, 0]),
                                paddle.ones_like(z[:, 0]))
            # print(pred)
            # print(label)
            acc_perm1 = (pred == label).astype("float32").mean()
            acc_perm2 = (pred == 1 - label).astype("float32").mean()
            acc = paddle.maximum(acc_perm1, acc_perm2)
            #print(acc)
            all_loss.append(*loss.numpy().tolist())
            all_acc.append(*acc.numpy().tolist())

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
        niters = len(all_loss)
        print("Epoch %d | loss %.4f | accuracy %.4f" %
              (i, sum(all_loss) / niters, sum(all_acc) / niters))


if __name__ == "__main__":
    main()
