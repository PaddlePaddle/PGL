# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import time
import datetime
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset

import pgl
from pgl.utils.logger import log


def time2obj(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    return data_sj


def time2int(time_sj):
    data_sj = time.strptime(time_sj, "%H:%M")
    time_int = int(time.mktime(data_sj))
    return time_int


def int2time(t):
    timestamp = datetime.datetime.fromtimestamp(t)
    return timestamp.strftime('"%H:%M"')


def func_add_t(x):
    time_strip = 600
    time_obj = time2obj(x)
    time_e = ((
        (time_obj.tm_sec + time_obj.tm_min * 60 + time_obj.tm_hour * 3600)) //
              time_strip) % 288
    return time_e


def func_add_h(x):
    time_obj = time2obj(x)
    hour_e = time_obj.tm_hour
    return hour_e


class PGL4WPFDataset(Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    15 days for training,
                        3 days for validation,
                        and 6 days for testing
    """

    def __init__(
            self,
            data_path,
            filename='wtb5_10.csv',
            flag='train',
            size=None,
            capacity=134,
            day_len=24 * 6,
            train_days=153,  # 15 days
            val_days=16,  # 3 days
            test_days=15,  # 6 days
            total_days=184,  # 30 days
            theta=0.9, ):

        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]

        self.start_col = 0
        self.capacity = capacity
        self.theta = theta

        # initialization
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.data_path = data_path
        self.filename = filename

        self.total_size = self.unit_size * total_days
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size

        self.test_size = test_days * self.unit_size
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        df_data, raw_df_data = self.data_preprocess(df_raw)
        self.df_data = df_data
        self.raw_df_data = raw_df_data

        data_x, graph = self.build_graph_data(df_data)
        log.info(f"data_shape: {data_x.shape}")
        log.info(f"graph: {graph}")
        self.data_x = data_x
        self.graph = graph

    def __getitem__(self, index):
        # Sliding window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = self.data_x[:, s_begin:s_end, :]
        seq_y = self.data_x[:, r_begin:r_end, :]

        if self.flag == "train":
            perm = np.arange(0, seq_x.shape[0])
            np.random.shuffle(perm)
            return seq_x[perm], seq_y[perm]
        else:
            return seq_x, seq_y

    def __len__(self):
        return self.data_x.shape[1] - self.input_len - self.output_len + 1

    def data_preprocess(self, df_data):

        feature_name = [
            n for n in df_data.columns
            if "Patv" not in n and 'Day' not in n and 'Tmstamp' not in n and
            'TurbID' not in n
        ]
        feature_name.append("Patv")

        new_df_data = df_data[feature_name]

        log.info('adding time')
        t = df_data['Tmstamp'].apply(func_add_t)
        new_df_data.insert(0, 'time', t)

        weekday = df_data['Day'].apply(lambda x: x % 7)
        new_df_data.insert(0, 'weekday', weekday)

        pd.set_option('mode.chained_assignment', None)
        raw_df_data = new_df_data
        new_df_data = new_df_data.replace(
            to_replace=np.nan, value=0, inplace=False)

        return new_df_data, raw_df_data

    def get_raw_df(self):
        return self.raw_df

    def build_graph_data(self, df_data):
        cols_data = df_data.columns[self.start_col:]
        df_data = df_data[cols_data]
        raw_df_data = self.raw_df_data[cols_data]

        data = df_data.values
        data = np.reshape(data,
                          [self.capacity, self.total_size, len(cols_data)])
        raw_data = raw_df_data.values
        raw_data = np.reshape(
            raw_data, [self.capacity, self.total_size, len(cols_data)])

        border1s = [
            0, self.train_size - self.input_len,
            self.train_size + self.val_size - self.input_len
        ]
        border2s = [
            self.train_size, self.train_size + self.val_size,
            self.train_size + self.val_size + self.test_size
        ]

        self.data_mean = np.expand_dims(
            np.mean(
                data[:, border1s[0]:border2s[0], 2:],
                axis=(1, 2),
                keepdims=True),
            0)
        self.data_scale = np.expand_dims(
            np.std(data[:, border1s[0]:border2s[0], 2:],
                   axis=(1, 2),
                   keepdims=True),
            0)

        #self.data_mean = np.mean(data[:, border1s[0]:border2s[0], 2:]).reshape([1, 1, 1, 1])
        #self.data_scale = np.std(data[:, border1s[0]:border2s[0], 2:]).reshape([1, 1, 1, 1])

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.raw_df = []
        for turb_id in range(self.capacity):
            self.raw_df.append(
                pd.DataFrame(
                    data=raw_data[turb_id, border1 + self.input_len:border2],
                    columns=cols_data))

        data_x = data[:, border1:border2, :]
        data_edge = data[:, border1s[0]:border2s[0], -1]
        edge_w = np.corrcoef(data_edge)

        k = 5
        topk_indices = np.argpartition(edge_w, -k, axis=1)[:, -k:]
        rows, _ = np.indices((edge_w.shape[0], k))
        kth_vals = edge_w[rows, topk_indices].min(axis=1).reshape([-1, 1])

        row, col = np.where(edge_w > kth_vals)
        edges = np.concatenate([row.reshape([-1, 1]), col.reshape([-1, 1])],
                               -1)
        graph = pgl.Graph(num_nodes=edge_w.shape[0], edges=edges)
        return data_x, graph


class TestPGL4WPFDataset(Dataset):
    """
    Desc: Data preprocessing,
    """

    def __init__(self, filename, capacity=134, day_len=24 * 6):

        super().__init__()
        self.unit_size = day_len

        self.start_col = 0
        self.capacity = capacity
        self.filename = filename

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.filename)
        df_data, raw_df_data = self.data_preprocess(df_raw)
        self.df_data = df_data
        self.raw_df_data = raw_df_data

        data_x = self.build_graph_data(df_data)
        self.data_x = data_x

    def data_preprocess(self, df_data):
        feature_name = [
            n for n in df_data.columns
            if "Patv" not in n and 'Day' not in n and 'Tmstamp' not in n and
            'TurbID' not in n
        ]
        feature_name.append("Patv")

        new_df_data = df_data[feature_name]

        log.info('adding time')
        t = df_data['Tmstamp'].apply(func_add_t)
        new_df_data.insert(0, 'time', t)

        weekday = df_data['Day'].apply(lambda x: x % 7)
        new_df_data.insert(0, 'weekday', weekday)

        pd.set_option('mode.chained_assignment', None)
        raw_df_data = new_df_data
        new_df_data = new_df_data.replace(to_replace=np.nan, value=0)

        return new_df_data, raw_df_data

    def get_raw_df(self):
        return self.raw_df

    def build_graph_data(self, df_data):
        cols_data = df_data.columns[self.start_col:]
        df_data = df_data[cols_data]
        raw_df_data = self.raw_df_data[cols_data]
        data = df_data.values
        raw_data = raw_df_data.values

        data = np.reshape(data, [self.capacity, -1, len(cols_data)])
        raw_data = np.reshape(raw_data, [self.capacity, -1, len(cols_data)])

        data_x = data[:, :, :]

        self.raw_df = []
        for turb_id in range(self.capacity):
            self.raw_df.append(
                pd.DataFrame(
                    data=raw_data[turb_id], columns=cols_data))
        return np.expand_dims(data_x, [0])

    def get_data(self):
        return self.data_x


if __name__ == "__main__":
    data_path = "./data"
    data = PGL4WPFDataset(data_path, filename="wtb5_10.csv")
