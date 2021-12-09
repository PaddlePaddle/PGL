import os
import os.path as osp
import glob
import pickle

import numpy as np
import pandas as pd

import sklearn
import sklearn.linear_model
import torch

def mae(pred, true):
    return np.mean(np.abs(pred-true))

model_root = "./model_pred"
max_min_drop_rate = 0.2     # <=0 means no drop, should < 0.5

split_idx = torch.load("../dataset/pcqm4m_kddcup2021/split_dict.pt")
cross_idx = pickle.load(open("../dataset/cross_split.pkl", 'rb'))
raw_df = pd.read_csv('../dataset/pcqm4m_kddcup2021/raw/data.csv.gz',
                     compression='gzip', sep=',')

model_cross_valid = glob.glob(osp.join(model_root, "**", 'crossvalid*'), recursive=True)
model_left_valid = [fname.replace('crossvalid', 'leftvalid') for fname in model_cross_valid]
model_test = [fname.replace('crossvalid', 'test') for fname in model_cross_valid]

def process(cross=1):
    model_path = []
    crossvalid_pred = []
    leftvalid_pred = []
    test_pred = []

    for i, model_p in enumerate(model_cross_valid):
        model_name = osp.dirname(model_p)
        if cross == 1 and model_name.endswith('cross2'):
            continue
        elif cross == 2 and model_name.endswith('cross1'):
            continue

        model_path.append(osp.splitext(model_p)[0])
        crossvalid_pred.append(np.load(model_p)['arr_0'])
        leftvalid_pred.append(np.load(model_left_valid[i])['arr_0'])
        test_pred.append(np.load(model_test[i])['arr_0'])

    return np.array(model_path).T, np.array(crossvalid_pred).T, \
           np.array(leftvalid_pred).T, np.array(test_pred).T

cross1_model_path, cross1_valid, cross1_left, cross1_test = process(cross=1)
cross2_model_path, cross2_valid, cross2_left, cross2_test = process(cross=2)

########################
def filter_model(pred, true):
    model = sklearn.linear_model.HuberRegressor(max_iter=10000)
    model.fit(pred, true)
    idx_trivial = np.abs(model.coef_*len(model.coef_)) < 1.8
    print("remaining #models:", np.sum(~idx_trivial))
    return idx_trivial

print("filtering cross1 with low contribution ...")
cross1_valid_true = raw_df['homolumogap'][cross_idx['cross_valid_1']]
cross1_trivial_idx = filter_model(cross1_valid, cross1_valid_true)
cross1_valid = cross1_valid[:, ~cross1_trivial_idx]

print("filtering cross2 with low contribution ...")
cross2_valid_true = raw_df['homolumogap'][cross_idx['cross_valid_2']]
cross2_trivial_idx = filter_model(cross2_valid, cross2_valid_true)
cross2_valid = cross2_valid[:, ~cross2_trivial_idx]

########################
if max_min_drop_rate > 0:
    drop_num = int(max_min_drop_rate * cross1_valid.shape[1])
    cross1_valid = np.sort(cross1_valid)[:, drop_num:-drop_num]
    cross2_valid = np.sort(cross2_valid)[:, drop_num:-drop_num]

cross1_model = sklearn.linear_model.HuberRegressor(max_iter=10000)
cross1_valid_true = raw_df['homolumogap'][cross_idx['cross_valid_1']]
print("fitting cross1 ...")
cross1_model.fit(cross1_valid, cross1_valid_true)

cross2_model = sklearn.linear_model.HuberRegressor(max_iter=10000)
cross2_valid_true = raw_df['homolumogap'][cross_idx['cross_valid_2']]
print("fitting cross2 ...")
cross2_model.fit(cross2_valid, cross2_valid_true)

########################
cross1_left = cross1_left[:, ~cross1_trivial_idx]
cross2_left = cross2_left[:, ~cross2_trivial_idx]
leftvalid_true = raw_df['homolumogap'][cross_idx['valid_left_1percent']]

drop_num = int(max_min_drop_rate * cross1_left.shape[1])

cross1_left = np.sort(cross1_left)[:, drop_num:-drop_num]
cross2_left = np.sort(cross2_left)[:, drop_num:-drop_num]
ensemble_left_pred = np.mean([cross1_model.predict(cross1_left),
                            cross2_model.predict(cross2_left)],
                        axis=0)
print("left valid mae:", mae(ensemble_left_pred, leftvalid_true))

cross1_test = cross1_test[:, ~cross1_trivial_idx]
cross2_test = cross2_test[:, ~cross2_trivial_idx]
cross1_test = np.sort(cross1_test)[:, drop_num:-drop_num]
cross2_test = np.sort(cross2_test)[:, drop_num:-drop_num]

########################
ensemble_test_pred = np.mean([cross1_model.predict(cross1_test),
                              cross2_model.predict(cross2_test)],
                             axis=0)

########################
test_smiles = raw_df['smiles'][split_idx['test']]

dic_known = {}
train_valid = np.append(split_idx['train'], split_idx['valid'])
smiles_list = raw_df['smiles']
value_list = raw_df['homolumogap'].values

for d in train_valid:
    smiles = smiles_list[d]
    value = value_list[d]
    if not smiles in dic_known:
        dic_known[smiles] = []
    dic_known[smiles].append(value)

for i, smiles in enumerate(test_smiles):
    if smiles in dic_known:
        ensemble_test_pred[i] = np.mean(dic_known[smiles])

########################
print('cross1 models:', cross1_model_path[~cross1_trivial_idx])
print('cross2 models:', cross2_model_path[~cross2_trivial_idx])
np.savez_compressed('y_pred_pcqm4m.npz',
                    y_pred=ensemble_test_pred.astype(np.float32))
