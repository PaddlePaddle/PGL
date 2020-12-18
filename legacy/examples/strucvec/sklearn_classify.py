"""
sklearn_classify.py
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
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random_seed = 67


def train_lr_l2_model(args, data):
    """
    The main function to train lr model with l2 regularization.
    """
    acc_list = []
    data = np.array(data)
    data = data[data[:, 0].argsort()]
    x_data = data[:, 1:-1]
    y_data = data[:, -1]
    for random_num in range(0, 10):
        X_train, X_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=0.2,
            random_state=random_num + random_seed)

        # use the one vs rest to train the lr model with l2 
        pred_test = []
        for i in range(0, args.num_class):
            y_train_relabel = np.where(y_train == i, 1, 0)
            y_test_relabel = np.where(y_test == i, 1, 0)
            lr = LogisticRegression(C=10.0, random_state=0, max_iter=100)
            lr.fit(X_train, y_train_relabel)
            pred = lr.predict_proba(X_test)
            pred_test.append(pred[:, -1].tolist())
        pred_test = np.array(pred_test)
        pred_test = np.transpose(pred_test)
        c_index = np.argmax(pred_test, axis=1)
        acc = accuracy_score(y_test.flatten(), c_index)
        acc_list.append(acc)
        print("pass:{}-acc:{}".format(random_num, acc))
    print("the avg acc is {}".format(np.mean(acc_list)))
