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

import paddle
from tqdm import tqdm
import pgl
import numpy as np
import torch
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
import time
import sys
import os

root = "dataset_path"
labels = np.load(
    os.path.join(root, "mag240_kddcup2021", "processed", "paper",
                 "node_label.npy"),
    mmap_mode="r")
split = torch.load(os.path.join(root, "mag240_kddcup2021", "split_dict.pt"))

graph_path = os.path.join(root, "paper_coauthor_paper_symmetric_jc0.5")
graph = pgl.Graph.load(graph_path)

import numpy as np

fold_id = int(sys.argv[2])
model_name = sys.argv[3]
alpha = float(sys.argv[1])
save_model_name = model_name + "_diff%s" % alpha

try:
    os.makedirs("result/" + save_model_name)
except Exception as e:
    print(e)
    pass

valid_label = np.load("result/%s/all_eval_result.npy" % model_name)
test_label = np.load("result/%s/test_%s.npy" % (model_name, fold_id))

# prepare indgree
indegree = paddle.to_tensor(graph.indegree(), dtype="float32")
indegree = (1.0 / (indegree + 1)).reshape([-1, 1])


def aggr(batch, y, nxt_y, y0, alpha):
    pred = graph.predecessor(batch.numpy())
    self_label = paddle.to_tensor(y[batch.numpy()])
    self_label0 = paddle.to_tensor(y0[batch.numpy()])
    pred_id = []
    for n, p in enumerate(pred):
        if len(p) > 0:
            pred_id.append(np.ones(len(p)) * n)
    pred_cat = np.concatenate(pred)
    pred_id_cat = paddle.to_tensor(np.concatenate(pred_id), dtype="int64")
    pred_cat_pd = paddle.to_tensor(pred_cat)

    pred_label = paddle.to_tensor(y[pred_cat])

    pred_norm = paddle.gather(indegree, pred_cat_pd)
    self_norm = paddle.gather(indegree, paddle.to_tensor(batch, dtype="int64"))

    others = paddle.zeros_like(self_label)
    others = paddle.scatter(others, pred_id_cat, pred_label)
    others = (1 - alpha) * (others + self_label
                            ) * self_norm + alpha * self_label0
    others = others / paddle.sum(others, -1, keepdim=True)
    nxt_y[batch] = others.numpy()


# prepare labels
N = graph.num_nodes
C = 153
y = np.zeros((N, C), dtype="float32")
y0 = np.zeros((N, C), dtype="float32")
nxt_y = np.zeros((N, C), dtype="float32")

train_idx = split['train'].tolist()
val_idx = split["valid"].tolist()
test_idx = split["test"].tolist()
y0[val_idx] = valid_label
y0[test_idx] = test_label

y[val_idx] = valid_label
y[test_idx] = test_label

for i in range(5):
    if i == fold_id:
        continue
    train_idx.extend(
        np.load("result/" + model_name + "/valid_%s.npy" % i).tolist())
train_idx = np.array(train_idx, dtype="int32")

val_idx = np.load("result/" + model_name + "/valid_%s.npy" % fold_id)
test_idx = split['test']

# set gold label

y[train_idx].fill(0)
y[train_idx, labels[train_idx].astype("int32")] = 1

y0[train_idx].fill(0)
y0[train_idx, labels[train_idx].astype("int32")] = 1


def smooth(y0, y, nxt_y, alpha=0.2):
    nodes = train_idx.tolist() + val_idx.tolist() + test_idx.tolist()
    pbar = tqdm(total=len(nodes))
    batch_size = 50000
    batch_no = 0
    nxt_y.fill(0)

    while batch_no < len(nodes):
        batch = nodes[batch_no:batch_no + batch_size]
        batch = paddle.to_tensor(batch, dtype="int64")
        aggr(batch, y, nxt_y, y0, alpha)
        batch_no += batch_size
        pbar.update(batch_size)


evaluator = MAG240MEvaluator()

best_acc = 0
hop = 0

train_label = labels[train_idx]
train_pred = y[train_idx]
train_pred = np.argmax(train_pred, -1)
train_acc = evaluator.eval({
    'y_true': train_label,
    'y_pred': train_pred
})['acc']
print("Hop", hop, "alpha", alpha, "Train Acc", train_acc)

valid_label = labels[val_idx]
valid_pred = y[val_idx]
valid_pred = np.argmax(valid_pred, -1)
valid_acc = evaluator.eval({
    'y_true': valid_label,
    'y_pred': valid_pred
})['acc']
print("Hop", hop, "alpha", alpha, "Valid Acc", valid_acc)

if valid_acc > best_acc:
    np.save("result/" + save_model_name + "/val_%s_pred.npy" % (fold_id),
            y[val_idx])
    np.save("result/" + save_model_name + "/test_%s.npy" % (fold_id),
            y[test_idx])
    np.save("result/" + save_model_name + "/valid_%s.npy" % (fold_id), val_idx)
    best_acc = valid_acc

for hop in range(1, 5):
    smooth(y0, y, nxt_y, alpha)
    nxt_y, y = y, nxt_y

    y[train_idx].fill(0)
    y[train_idx, labels[train_idx].astype("int32")] = 1

    train_label = labels[train_idx]
    train_pred = y[train_idx]
    train_pred = np.argmax(train_pred, -1)
    train_acc = evaluator.eval({
        'y_true': train_label,
        'y_pred': train_pred
    })['acc']
    print("Hop", hop, "alpha", alpha, "Train Acc", train_acc)

    valid_label = labels[val_idx]
    valid_pred = y[val_idx]
    valid_pred = np.argmax(valid_pred, -1)
    valid_acc = evaluator.eval({
        'y_true': valid_label,
        'y_pred': valid_pred
    })['acc']
    print("Hop", hop, "alpha", alpha, "Valid Acc", valid_acc)

    if valid_acc > best_acc:
        np.save("result/" + save_model_name + "/val_%s_pred.npy" % (fold_id),
                y[val_idx])
        np.save("result/" + save_model_name + "/test_%s.npy" % (fold_id),
                y[test_idx])
        np.save("result/" + save_model_name + "/valid_%s.npy" % (fold_id),
                val_idx)
        best_acc = valid_acc
