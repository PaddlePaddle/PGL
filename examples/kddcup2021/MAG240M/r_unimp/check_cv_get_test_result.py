import os
import yaml
import pgl
import time
import copy
import numpy as np
import os.path as osp
from pgl.utils.logger import log
from pgl.graph import Graph
from pgl import graph_kernel
from pgl.sampling.custom import subgraph
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from dataset.base_dataset import BaseDataGenerator
import time
from tqdm import tqdm
from easydict import EasyDict as edict
import argparse


def get_result(config, eval_all=False):
    dataset = MAG240MDataset(config.data_dir)
    evaluator = MAG240MEvaluator()
    file = 'model_result_temp'
    sudo_label = np.memmap(file, dtype=np.float32, mode='r',
                          shape=(121751666, 153))
    file = "ck_result.txt"
    wf = open(file, 'a', encoding='utf-8')
    label = dataset.all_paper_label
    if eval_all:
        valid_idx = dataset.get_idx_split('valid')
        pred = sudo_label[valid_idx]
        save_path = os.path.join(config.valid_path, "all_eval_result")
        np.save(save_path, pred)
        y_pred = pred.argmax(1)
        y_true = label[valid_idx]
        valid_acc = evaluator.eval({
                                'y_true': y_true,
                                'y_pred': y_pred
                            })['acc']
        print("all eval result\n")
        print(f"valid_acc: {valid_acc}\n")
        wf.write("all eval result\n")
        wf.write(f"valid_acc: {valid_acc}\n")
        
    else:
        valid_path = os.path.join(config.valid_path, config.valid_name)
        valid_idx = np.load(valid_path)
        test_idx = dataset.get_idx_split('test')
        
        pred = sudo_label[valid_idx]
        y_pred = pred.argmax(1)
        y_true = label[valid_idx]
        valid_acc = evaluator.eval({
                                'y_true': y_true,
                                'y_pred': y_pred
                            })['acc']
        print(f"eval cv {config.valid_name} result\n")
        print(f"valid_acc: {valid_acc}\n")
        wf.write(f"eval cv {config.valid_name} result\n")
        wf.write(f"valid_acc: {valid_acc}\n")
        
        save_path_test = os.path.join(config.valid_path, config.test_name)
        pred_test = sudo_label[test_idx]
        print(pred_test.shape)
        np.save(save_path_test, pred_test)
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument("--eval_all", action='store_true', default=False)
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.samples = [int(i) for i in config.samples.split('-')]

    print(config)
    get_result(config, args.eval_all)