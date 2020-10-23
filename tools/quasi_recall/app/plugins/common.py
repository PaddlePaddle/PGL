#!/usr/bin/env python
# coding=utf-8
"""
 @auth : wangna07@baidu.com
 @date : 2020-08-03 17:55:55
"""
import logging; log = logging.getLogger()
from ..platform import PRProcessor

@PRProcessor.plugin_register('common')
class QuasiRecallContext(object):
    def process(self, info):
        """calculate PR entry for recall model"""
        self.info = info
        if self.info.get("recall") is None and self.info.get("truth") is None:
            log.info("The case is not suitable for plugin common.")
            return {}, False
        log.info(f"start to execute plugin: {info['plugins_type']}")
        self.initialize()
        self.load_truth_list()
        self.load_recall_list()
        res, status = self.cal_quasi_recall()
        return res, status

    def initialize(self):
        """initialize var"""
        pass

    def load_truth_list(self):
        """load truth list"""
        flag = isinstance(self.info['truth'], dict)
        if flag == False: self.info['truth'] = eval(self.info['truth'])

    def load_recall_list(self):
        """load recall list""" 
        flag = isinstance(self.info['recall'], dict)
        if flag == False: self.info['recall'] = eval(self.info['recall'])
    
    def cal_quasi_recall(self):
        """calculate PR"""
        res = dict()
        log.info(f"{self.info['plan_type']}: start to cal quasi recall")
        truth_ratios, recall_ratios = list(), list()

        for (key, recall_list) in self.info['recall'].items():
            if self.info['truth'].get(key):
                truth_list = self.info['truth'].get(key)
                recall_truth_list = [i for i in truth_list if i in set(recall_list)]
                truth_ratios.append(len(recall_truth_list) / len(recall_list))
                recall_ratios.append(len(recall_truth_list) / len(truth_list))
        truth_ratio = sum(truth_ratios) / len(truth_ratios) if truth_ratios else 0
        recall_ratio = sum(recall_ratios) / len(recall_ratios) if recall_ratios else 0
        log.info(f"{self.info['plan_type']}: truth_ratio={truth_ratio} recall_ratio={recall_ratio}")
        res["recall_ratio"] = recall_ratio
        res["truth_ratio"] = truth_ratio
        status = True
        return res, status
