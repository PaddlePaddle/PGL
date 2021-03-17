"""
classify.py
"""
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
import numpy as np
import paddle
import paddle.fluid as fluid


def build_lr_model(args):
    """
    Build the LR model to train.
    """
    emb_x = fluid.layers.data(
        name="emb_x", dtype='float32', shape=[args.w2v_emb_size])
    label = fluid.layers.data(name="label_y", dtype='int64', shape=[1])
    logits = fluid.layers.fc(input=emb_x,
                             size=args.num_class,
                             act=None,
                             name='classification_layer')
    proba = fluid.layers.softmax(logits)
    loss = fluid.layers.softmax_with_cross_entropy(logits, label)
    loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=proba, label=label, k=1)
    return loss, acc


def construct_feed_data(data):
    """
    Construct the data to feed model.
    """
    datas = []
    labels = []
    for sample in data:
        if len(datas) < 16:
            labels.append([sample[-1]])
            datas.append(sample[1:-1])
        else:
            yield np.array(datas).astype(np.float32), np.array(labels).astype(
                np.int64)
            datas = []
            labels = []
    if len(datas) != 0:
        yield np.array(datas).astype(np.float32), np.array(labels).astype(
            np.int64)


def run_epoch(exe, data, program, stage, epoch, loss, acc):
    """
    The epoch funtcion to run each epoch.
    """
    print('start {} epoch of {}'.format(stage, epoch))
    all_loss = 0.0
    all_acc = 0.0
    all_samples = 0.0
    count = 0
    for datas, labels in construct_feed_data(data):
        batch_loss, batch_acc = exe.run(
            program,
            fetch_list=[loss, acc],
            feed={"emb_x": datas,
                  "label_y": labels})
        len_samples = len(datas)
        all_loss = batch_loss * len_samples
        all_acc = batch_acc * len_samples
        all_samples += len_samples
        count += 1
    print("pass:{}, epoch:{}, loss:{}, acc:{}".format(stage, epoch, batch_loss,
                                                      all_acc / (len_samples)))


def train_lr_model(args, data):
    """
    The main function to run the lr model.
    """
    data_nums = len(data)
    train_data_nums = int(0.8 * data_nums)
    train_data = data[:train_data_nums]
    test_data = data[train_data_nums:]

    place = fluid.CPUPlace()

    train_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        loss, acc = build_lr_model(args)
    test_program = train_program.clone(for_test=True)

    with fluid.program_guard(train_program, startup_program):
        adam = fluid.optimizer.Adam(learning_rate=args.lr)
        adam.minimize(loss)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    for epoch in range(0, args.epoch):
        run_epoch(exe, train_data, train_program, "train", epoch, loss, acc)
        print('-------------------')
        run_epoch(exe, test_data, test_program, "valid", epoch, loss, acc)
