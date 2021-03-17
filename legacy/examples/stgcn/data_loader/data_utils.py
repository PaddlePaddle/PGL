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
"""data processing
"""
import numpy as np
import pandas as pd

from utils.math_utils import z_score


class Dataset(object):
    """Dataset
    """

    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):  # type: train, val or test
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    """Generate data in the form of standard sequence unit."""
    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(
                data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq


def adj_matrx_gen_custom(input_file, city_file):
    """genenrate Adjacency Matrix from file 
    """
    print("generate adj_matrix data (take long time)...")
    # data
    df = pd.read_csv(
        input_file,
        sep='\t',
        names=['date', '迁出省份', '迁出城市', '迁入省份', '迁入城市', '人数'])
    # 只需要2020年的数据
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
    df = df.set_index('date')
    df = df['2020']
    city_df = pd.read_csv(city_file)
    # 剔除武汉
    city_df = city_df.drop(0)
    num = len(city_df)
    matrix = np.zeros([num, num])
    for i in city_df['city']:
        for j in city_df['city']:
            if (i == j):
                continue
            # 选出从i到j的每日人数
            cut = df[df['迁出城市'].str.contains(i)]
            cut = cut[cut['迁入城市'].str.contains(j)]
            # 求均值作为权重
            average = cut['人数'].mean()
            # 赋值给matrix
            i_index = int(city_df[city_df['city'] == i]['num']) - 1
            j_index = int(city_df[city_df['city'] == j]['num']) - 1
            matrix[i_index, j_index] = average

    np.savetxt("dataset/W_74.csv", matrix, delimiter=",")


def data_gen_custom(input_file, output_file, city_file, n, n_his, n_pred,
                    n_config):
    """data_gen_custom"""
    print("generate training data...")
    # data
    df = pd.read_csv(
        input_file,
        sep='\t',
        names=['date', '迁出省份', '迁出城市', '迁入省份', '迁入城市', '人数'])
    # 只需要2020年的数据
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
    df = df.set_index('date')
    df = df['2020']
    city_df = pd.read_csv(city_file)
    input_df = pd.DataFrame()

    out_df_wuhan = df[df['迁出城市'].str.contains('武汉')]
    for i in city_df['city']:
        # 筛选迁入城市
        in_df_i = out_df_wuhan[out_df_wuhan['迁入城市'].str.contains(i)]
        # 确保按时间升序
        # in_df_i.sort_values("date",inplace=True)
        # 按时间插入
        in_df_i.reset_index(drop=True, inplace=True)
        input_df[i] = in_df_i['人数']

    # 替换Nan值
    input_df = input_df.replace(np.nan, 0)

    x = input_df
    y = pd.read_csv(output_file)
    # 删除第1列
    x.drop(
        x.columns[x.columns.str.contains(
            'unnamed', case=False)],
        axis=1,
        inplace=True)
    y = y.drop(columns=['date'])

    # 剔除迁入武汉的数据
    x = x.drop(columns=['武汉'])
    y = y.drop(columns=['武汉'])

    # param
    n_val, n_test = n_config
    n_train = len(y) - n_val - n_test - 2

    # (?,26,74,1)
    df = pd.DataFrame(columns=x.columns)
    for i in range(len(y) - n_pred + 1):
        df = df.append(x[i:i + n_his])
        df = df.append(y[i:i + n_pred])
    data = df.values.reshape(-1, n_his + n_pred, n,
                             1)  # n == num_nodes == city num

    x_stats = {'mean': np.mean(data), 'std': np.std(data)}

    x_train = data[:n_train]
    x_val = data[n_train:n_train + n_val]
    x_test = data[n_train + n_val:]

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    print("generate successfully!")

    return dataset


def data_gen_mydata(input_file, label_file, n, n_his, n_pred, n_config):
    """data processing
    """
    # data
    x = pd.read_csv(input_file)
    y = pd.read_csv(label_file)
    x = x.drop(columns=['date'])
    y = y.drop(columns=['date'])

    # param
    n_val, n_test = n_config
    n_train = len(y) - n_val - n_test - 2

    # (?,26,74,1)
    df = pd.DataFrame(columns=x.columns)
    for i in range(len(y) - n_pred + 1):
        df = df.append(x[i:i + n_his])
        df = df.append(y[i:i + n_pred])

    data = df.values.reshape(-1, n_his + n_pred, n, 1)

    x_stats = {'mean': np.mean(data), 'std': np.std(data)}

    x_train = data[:n_train]
    x_val = data[n_train:n_train + n_val]
    x_test = data[n_train + n_val:]

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def data_gen(file_path, data_config, n_route, n_frame=21, day_slot=288):
    """Source file load and dataset generation."""
    n_train, n_val, n_test = data_config
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route,
                       day_slot)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    """Data iterator in batch.

    Args:
        inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
        batch_size: int, size of batch.
        dynamic_batch: bool, whether changes the batch size in the last batch 
            if its length is less than the default.
        shuffle: bool, whether shuffle the batches.
    """
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]
