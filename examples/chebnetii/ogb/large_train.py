# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random
import argparse
import uuid
import numpy as np
import paddle
from paddle.optimizer import Adam
from model import ChebNetII
from utils import set_seed

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument(
    '--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--data', type=str, default="papers100m", help='datasets.')

parser.add_argument('--net', type=str, default="ChebNetII", help='device id')
parser.add_argument('--batch_size', type=int, default=10000, help='Batch size')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument(
    '--weight_decay', type=float, default=0.0, help='weight decay.')
parser.add_argument(
    '--early_stopping', type=int, default=300, help='early stopping.')
parser.add_argument('--hidden', type=int, default=2048, help='hidden units.')
parser.add_argument(
    '--dropout', type=float, default=0.5, help='dropout for neural networks.')

parser.add_argument('--K', type=int, default=10, help='propagation steps.')
parser.add_argument(
    '--pro_lr',
    type=float,
    default=0.01,
    help='learning rate for BernNet propagation layer.')
parser.add_argument(
    '--pro_wd',
    type=float,
    default=0.00005,
    help='learning rate for BernNet propagation layer.')
args = parser.parse_args()
print(args)
set_seed(args.seed)

batch_size = args.batch_size
name = args.data
data_path = './data/'
train_data = np.load(data_path + "training_" + name + ".npy")
valid_data = np.load(data_path + "validation_" + name + ".npy")
test_data = np.load(data_path + "test_" + name + ".npy")

train_data = [paddle.to_tensor(mat) for mat in train_data[:args.K + 1]]
valid_data = [paddle.to_tensor(mat) for mat in valid_data[:args.K + 1]]
test_data = [paddle.to_tensor(mat) for mat in test_data[:args.K + 1]]

labels = np.load(data_path + "labels_" + name + ".npz")
train_labels = paddle.to_tensor(np.expand_dims(labels['tr_lb'], -1))
valid_labels = paddle.to_tensor(np.expand_dims(labels['va_lb'], -1))
test_labels = paddle.to_tensor(np.expand_dims(labels['te_lb'], -1))
print(train_labels.shape)

num_features = train_data[0].shape[1]
num_labels = int(train_labels.max()) + 1
print("Number of labels for " + name, num_labels)

checkpt_file = './pretrained/' + uuid.uuid4().hex + '.pdparams'
print(checkpt_file)

#breakpoint()
criterion = paddle.nn.loss.CrossEntropyLoss()
model = ChebNetII(num_features, args.hidden, num_labels, args)
optimizer = Adam(
    learning_rate=args.lr,
    parameters=[{
        'params': model.lin1.parameters()
    }, {
        'params': model.lin2.parameters()
    }, {
        'params': model.lin3.parameters()
    }, {
        'params': model.temp,
        'weight_decay': args.pro_wd,
        'learning_rate': args.pro_lr,
    }],
    weight_decay=args.weight_decay)


def create_batch(input_data):
    num_sample = input_data[0].shape[0]
    list_bat = []
    for i in range(0, num_sample, batch_size):
        if (i + batch_size) < num_sample:
            list_bat.append((i, i + batch_size))
        else:
            list_bat.append((i, num_sample))
    return list_bat


def train(st, end):
    model.train()
    output = model(train_data, st, end)
    acc_train = paddle.metric.accuracy(
        input=output, label=train_labels[st:end], k=1)
    loss_train = criterion(output, train_labels[st:end])
    loss_train.backward()
    optimizer.step()
    optimizer.clear_grad()

    return loss_train.item(), acc_train.item()


@paddle.no_grad()
def validate(st, end):
    model.eval()
    output = model(valid_data, st, end)
    loss_val = criterion(output, valid_labels[st:end])
    acc_val = paddle.metric.accuracy(
        input=output, label=valid_labels[st:end], k=1)
    return loss_val.item(), acc_val.item()


@paddle.no_grad()
def test(st, end):
    model.eval()
    output = model(test_data, st, end)
    loss_test = criterion(output, test_labels[st:end])
    acc_test = paddle.metric.accuracy(
        input=output, label=test_labels[st:end], k=1)
    return loss_test.item(), acc_test.item()


list_bat_train = create_batch(train_data)
list_bat_val = create_batch(valid_data)
list_bat_test = create_batch(test_data)

bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
valid_num = valid_data[0].shape[0]
test_num = test_data[0].shape[0]
for epoch in range(args.epochs):
    list_loss = []
    list_acc = []
    random.shuffle(list_bat_train)
    for st, end in list_bat_train:
        loss_tra, acc_tra = train(st, end)
        list_loss.append(loss_tra)
        list_acc.append(acc_tra)
    loss_tra = np.round(np.mean(list_loss), 4)
    acc_tra = np.round(np.mean(list_acc), 4)

    list_loss_val = []
    list_acc_val = []
    for st, end in list_bat_val:
        loss_val, acc_val = validate(st, end)
        list_loss_val.append(loss_val)
        list_acc_val.append(acc_val)

    loss_val = np.round(np.mean(list_loss_val), 4)
    acc_val = np.round(np.mean(list_acc_val), 4)

    if epoch % 5 == 0:
        print('train_acc:', acc_tra, '>>>>>>>>>>train_loss:', loss_tra)
        print('val_acc:', acc_val, '>>>>>>>>>>>val_loss:', loss_val)

    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        paddle.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.early_stopping:
        break

list_loss_test = []
list_acc_test = []
model.set_state_dict(paddle.load(checkpt_file))
for st, end in list_bat_test:
    loss_test, acc_test = test(st, end)
    list_loss_test.append(loss_test)
    list_acc_test.append(acc_test)
acc_test = np.round(np.mean(list_acc_test), 4)

print(name)
print('Load {}th epoch'.format(best_epoch))
print(
    f"Valdiation accuracy: {np.round(acc*100,2)}, Test accuracy: {np.round(acc_test*100,2)}"
)
