#!/usr/bin/env python
# coding=utf-8
"""
 @auth : wangna07@baidu.com
 @date : 2020-08-03 16:32:52
"""

from .platform import PRProcessor
import logging; log = logging.getLogger()
import json

class MainEntry(object):
    """main entrance"""
    def __init__(self, config, args):
        self._case_data = config.case_data
        self._new_item_data = config.new_item_data
        self._base_item_data = config.base_item_data
        self._new_user_data = config.new_user_data
        self._base_user_data = config.base_user_data
        self._data_type = config.data_type
        self._user_time = config.user_time
        self._item_time = config.item_time
        self._plan_type = config.plan_type
        self._plugins_type = config.plugins_type
        self._max_records = config.max_records
        self._job_name = config.jobname
        self._dim = config.dim
        self._n_trees = config.n_trees
        self._search_n = config.search_n
        self._ugi = config.ugi
        self._afs_conf = config.afs_conf
        self._truth = args.truth
        self._recall = args.recall
        self._tmp_dir = args.tmp_dir

    def cal_test(self):
        """init variable"""
        processor = PRProcessor()
        params = {}
        params["case_data"] = self._case_data
        params["new_item_data"] = self._new_item_data
        params["base_item_data"] = self._base_item_data
        params["new_user_data"] = self._new_user_data
        params["base_user_data"] = self._base_user_data
        params["data_type"] = self._data_type
        params["user_time"] = self._user_time
        params["item_time"] = self._item_time
        params["plan_type"] = self._plan_type
        params["dim"] = self._dim
        params["n_trees"] = self._n_trees
        params["search_n"] = self._search_n
        params["truth"] = self._truth
        params["recall"] = self._recall
        params["plugins_type"] = self._plugins_type
        params["job_name"] = self._job_name
        params["tmp_dir"] = self._tmp_dir
        params["ugi"] = self._ugi
        params["afs_conf"] = self._afs_conf
        params["max_records"] = self._max_records
        params["case_data_list"] = list()
        params["idxKey"] = dict()
        params["keyIdx"] = dict()
        params["annoy"] = None
        params["cli"] = None
        # used to choose shared case data
        params["case_dict"] = dict()
        # used to judge evalutae test type(multi or single)
        params["vec_dict"] = dict()
        # default setting to single
        params["diff_type"] = "single"

        log.info(f"[Adopt plugin]:{self._plugins_type}")
        result, status = processor.process(params, plugins=(self._plugins_type,))
        log.info(f"{result}")
        if status == False: return {}, False
        return result, status


