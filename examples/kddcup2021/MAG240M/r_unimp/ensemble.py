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

import glob
import os
import pandas as pd
import numpy as np
import json
import tqdm
from collections import Counter

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
evaluator = MAG240MEvaluator()

import torch

root = "dataset_path"

split = torch.load(os.path.join(root, "mag240m_kddcup2021", "split_dict.pt"))
labels = np.load(
    os.path.join(root, "mag240m_kddcup2021", "processed", "paper",
                 "node_label.npy"),
    mmap_mode="r")

val_idx = split['valid']


class ModelStatus(object):
    def __init__(self, path):
        self.path = path.split("/")[-1]
        fold = 5
        self.val_pred = np.load(os.path.join(path, "all_eval_result.npy"), "r")
        self.test_pred = 0
        self.all_test = []
        for i in range(fold):
            self.test_pred = self.test_pred + np.load(
                os.path.join(path, "test_%s.npy" % i), "r")
            self.all_test.append(
                np.argmax(np.load(os.path.join(path, "test_%s.npy" % i)), -1))
        self.test_pred = self.test_pred / fold

        valid_label = labels[val_idx]
        valid_pred = np.argmax(self.val_pred, -1)
        self.val_acc = evaluator.eval({
            'y_true': valid_label,
            'y_pred': valid_pred
        })['acc']

    def print(self):
        corr = np.ones(
            (len(self.all_test), len(self.all_test)), dtype="float32")
        for i, test_i in enumerate(self.all_test):
            for j, test_j in enumerate(self.all_test[:i]):
                score = np.mean(test_i == test_j)
                corr[i, j] = score
                corr[j, i] = score

        ret = {"path": self.path.split("/")[-1], }
        ret["val_acc"] = self.val_acc
        print(json.dumps(ret, indent=2))
        print("corr")
        print(corr)
        return ret


class EnsembleModelStatus(ModelStatus):
    def __init__(self, models):
        self.paths = []
        for model in models:
            if hasattr(model, "paths"):
                self.paths.extend(model.paths)
            if hasattr(model, "path"):
                self.paths.append(model.path)

        self.val_pred = models[0].val_pred
        for model in models[1:]:
            self.val_pred = self.val_pred + model.val_pred
        self.val_pred /= len(models)

        self.test_pred = models[0].test_pred
        for model in models[1:]:
            self.test_pred = self.test_pred + model.test_pred
        self.test_pred /= len(models)

        valid_label = labels[val_idx]
        valid_pred = np.argmax(self.val_pred, -1)
        self.val_acc = evaluator.eval({
            'y_true': valid_label,
            'y_pred': valid_pred
        })['acc']

    def print(self):
        ret = {"paths": self.paths}
        ret["val_acc"] = self.val_acc
        print(json.dumps(ret, indent=2))
        return ret


class EnsembleModels(object):
    def __init__(self, models, beam_size=16):
        self.based_models = {m.path: m for m in models}
        self.beam_size = beam_size

        self.start_models = sorted(models, key=lambda x: -x.val_acc)
        self.start_models = [EnsembleModelStatus(self.start_models)]
        self.best_model_paths = self.start_models[0].paths

        self.best_val_acc = self.start_models[0].val_acc
        self.best_test_pred = self.start_models[0].test_pred
        print("Before Ensemble Best", "Cross Full", self.best_val_acc)

    def step(self, ):
        new_models = []
        for last_m in tqdm.tqdm(self.start_models):
            for key, based_m in self.based_models.items():
                if hasattr(last_m, "path"):
                    paths = [last_m.path, key]
                else:
                    paths = last_m.paths + [key]
                enm = [self.based_models[m] for m in paths]
                new_models.append(EnsembleModelStatus(enm))
        new_models = sorted(
            new_models, key=lambda x: -x.val_acc)[:self.beam_size]
        self.start_models = new_models
        if self.best_val_acc < self.start_models[0].val_acc:
            self.best_val_acc = self.start_models[0].val_acc
            self.best_model_paths = self.start_models[0].paths
            print("Find Better", "Cross Full", self.best_val_acc)
            print("Model Diff",
                  np.mean(
                      np.argmax(self.best_test_pred, -1) == np.argmax(
                          self.start_models[0].test_pred, -1)))
            print("Model Diff Num",
                  np.sum(
                      np.argmax(self.best_test_pred, -1) == np.argmax(
                          self.start_models[0].test_pred, -1)))
            self.best_test_pred = self.start_models[0].test_pred
            np.save("best_result.npy", self.start_models[0].test_pred)


def get_corr(m_i, m_j):
    return np.mean(np.argmax(m_i, -1) == np.argmax(m_j, -1))


def find_models(paths):
    all_models_path = []
    for path in paths:
        all_models_path.extend(glob.glob(os.path.join(path, "*")))
    print(all_models_path)
    all_models = []

    for model in all_models_path:
        model = ModelStatus(model)
        all_models.append(model)
        model.print()

    for i, m_i in enumerate(all_models):
        for j, m_j in enumerate(all_models[:i]):
            print(m_i.path, m_j.path, get_corr(m_i.test_pred, m_j.test_pred))

    em = EnsembleModels(all_models)
    for i in range(10):
        em.step()
    count = Counter()
    for path in em.best_model_paths:
        count[path] += 1

    for model in all_models:
        print(model.path, model.val_acc)

    for key, value in count.items():
        print(key, value)
    print("Best-Cross Full", em.best_val_acc)


if __name__ == "__main__":
    # automatic ensemble from result dir
    model_paths = ["result"]
    find_models(model_paths)
    y_pred = np.load("best_result.npy")
    # save ensemble result
    input_dict = {'y_pred': y_pred}
    evaluator.save_test_submission(input_dict=input_dict, dir_path="./")
    print("Save success")
