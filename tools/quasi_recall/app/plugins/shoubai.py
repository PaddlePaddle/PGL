#!/usr/bin/env python
# coding=utf-8
"""
 @auth : wangna07@baidu.com
 @date : 2020-08-03 17:56:39
"""

import logging; log = logging.getLogger()
from ..platform import PRProcessor
import os
import ujson
import json
from subprocess import Popen, PIPE, DEVNULL

@PRProcessor.plugin_register('shoubai')
class QuasiRecallContext(object):
    def process(self, info):
        """calculate PR entry for recall model"""
        self.info = info
        recall = info.get('recall')
        truth = info.get('truth')
        if info.get("recall") is not None and info.get("truth") is not None:
            log.info(f"The case is not suitable for plugin shoubai.")
            return {}, False
        self.info['recall'] = dict()
        self.info['truth'] = dict()
        log.info(f"start to execute plugin: {info['plugins_type']}")
        self.initialize()
        res, status = self.task_main_entry_new(self.info['diff_type'])
        return res, status

    def task_main_entry_new(self, execute_type):
        """task execute main entry"""
        from operator import itemgetter
        from concurrent.futures import ThreadPoolExecutor, wait, as_completed, ALL_COMPLETED
        log.info(f"{execute_type}")
        result = {'recall_ratio': {}, 'truth_ratio': {}}
        def thread_func(key, vector_item):
            """thread execute function"""
            log.info(f"vector key: {key}")
            self.build_annoy_new(key)
            self.load_truth_list(key)
            self.load_recall_list(key)
            res, status = self.cal_quasi_recall(key)
            return (res, key)
        thread_pool, tasks = ThreadPoolExecutor(max_workers=5), list()
        for key, item in self.info['vec_dict'].items():
            tasks.append(thread_pool.submit(thread_func, key, item))
            
        log.info(f"finished")
        for (res, key) in [t.result() for t in as_completed(tasks)]:
            if key not in result['recall_ratio'] or key not in result['truth_ratio']:
                result['recall_ratio'].update(res['recall_ratio'])
                result['truth_ratio'].update(res['truth_ratio'])
        for idx, t in enumerate(as_completed(tasks)):
            log.info(f"{len(tasks)}, {idx + 1}")
        status = True
        return result, status

    def input_data_ready(self):
        """input parameters download"""
        log.info(f"start to download related input data")
        data = ""
        plan_type = self.info.get('plan_type')

        if data := self.info.get('case_data'):
            if 'ftp' in data: self.ftp_download_task('case_data')
            elif 'hdfs' in data: self.hdfs_single_download_task('case_data')

        if plan_type != "icf":
            if data := self.info.get('new_user_data'):  
               if 'ftp' in data: self.ftp_download_task('new_user_data')
               elif 'hdfs' in data: self.hdfs_single_download_task('new_user_data')

            if data := self.info.get('base_user_data'):  
               if 'ftp' in data: self.ftp_download_task('base_user_data')
               elif 'hdfs' in data: self.hdfs_single_download_task('base_user_data')
            
        elif plan_type != "ucf":
            if data := self.info.get('new_item_data'):
                if 'ftp' in data: self.ftp_download_task('new_item_data')
                elif 'hdfs' in data: self.hdfs_single_download_task('new_item_data')

            if data := self.info.get('base_item_data'):  
               if 'ftp' in data: self.ftp_download_task('base_item_data')
               elif 'hdfs' in data: self.hdfs_single_download_task('base_item_data')

    def filter_case_data(self):
        """filter case data by user vector"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def single_filter_func(vector_list):
            """single vector thread execute function"""
            for vec_item in vector_list:
                key, *vector = vec_item.strip().split()
                if key not in self.info['case_data_list']: self.info['case_data_list'].append(key)
                if len(self.info['case_data_list']) == 5000: break
            log.info(f"single-keys_list:{len(self.info['case_data_list'])}")
            return (self.info['case_data_list'])

        def multi_filter_func(vector_list, vector_key):
            """multi vector thread execute function"""
            log.info(f"start to filter {vector_key}")
            if vector_key not in self.info["case_dict"]: self.info["case_dict"][vector_key] = list()
            check_key = lambda key: key not in self.info["case_dict"][vector_key]
            self.info["case_dict"][vector_key] = [vec_item.strip().split()[0] for vec_item in vector_list if check_key(vec_item.strip().split()[0])]
            log.info(f"multi-keys_list:{len(self.info['case_dict'][vector_key])}")
            return (self.info['case_dict'][vector_key], vector_key)

        thread_pool, tasks = ThreadPoolExecutor(max_workers=8), list()
        for vector_key in self.info['vec_dict'].keys():
            if len(self.info['case_data_list']) == 0 and self.info['diff_type'] == "single":
                tasks.append(thread_pool.submit(single_filter_func, self.info["vec_dict"][vector_key]))
            elif len(self.info['case_data_list']) == 0 and self.info['diff_type'] == "multi":
                log.info("start to append case key")
                tasks.append(thread_pool.submit(multi_filter_func, self.info["vec_dict"][vector_key], vector_key))
                log.info("end to append case key")
            else:
                log.info("no need to filter case data")

        if len(self.info['case_data_list']) == 0 and self.info['diff_type'] == "multi":
            for (res_list, key) in [t.result() for t in as_completed(tasks)]:
                """need to compare two vector keys"""
                log.info(f"case load success: {key}")

            set1 = set(self.info['case_dict']['new_user_data'])
            set2 = set(self.info['case_dict']['base_user_data'])
            iset = set1.intersection(set2)
            self.info['case_data_list'] = [item for item in iset][:5000]
            log.info(f"{len(self.info['case_data_list'])}")
            log.info(f"{self.info['case_data_list']}")
        else:
            log.info(f"no need to process case data")

        for idx, t in enumerate(as_completed(tasks)):
            log.info(f"{len(tasks)}, {idx + 1}")
        status = True

    def check_vector_info(self):
        """Check the number of experiments to be evaluated"""
        count_test = len(self.info['vec_dict'].keys())
        if count_test >= 2 and "base_user_data" in self.info['vec_dict'].keys():
            log.info("evaluate multi-group tests")
            self.info['diff_type'] = "multi"
        else:
            log.info("only to evaluate one test")
            self.info['diff_type'] = "single"

    def ftp_download_task(self, data_key):
        """using ftp to download"""
        tmp_dir = self.info.get('tmp_dir') 
        data = os.path.join(tmp_dir, data_key)
        if data_ftp := self.info.get(data_key):
            cmd = ["wget", "-q", "-O", data, data_ftp]
            log.info(f"{cmd}")
            if Popen(cmd, stdout=DEVNULL, stderr=DEVNULL).wait():
                 err_msg=f"{data_ftp} download failed, please check."
                 log.error(err_msg)
                 print(ujson.dumps(dict(err_no=1, err_msg=err_msg)))
                 exit()
            with open(data) as f:
                if data_key == 'case_data':
                    for (lineno, line) in enumerate(f, 1):
                        if lineno > 5000: break
                        self.info['case_data_list'].append(line.strip().split()[0])
                    log.info(f"{self.info['data_type']}load case data{len(self.info['case_data_list'])}")
                else:
                    vector_list = list()
                    for (lineno, line) in enumerate(f, 1):
                        if lineno > self.info.get('max_records'): break
                        vector_list.append(line.strip())
                    self.info[data_key] = vector_list
                    log.info(f"{self.info['data_type']}load vector data{len(self.info[data_key])}")
                    self.info['vec_dict'][data_key] = self.info[data_key]

    def hdfs_single_download_task(self, data_key):
        """using hdfs to single-download"""
        if data_hdfs := self.info.get(data_key):
            ugi = self.info.get('ugi')
            afs_conf = self.info.get('afs_conf')
            cmd = ["hadoop", "fs", "-conf", f"{afs_conf}",
                   "-D", "hadoop.job.ugi="f"{ugi}",
                   "-ls", f"{data_hdfs}"]
            hdfs = Popen(cmd, stdout=PIPE, stderr=DEVNULL)
            part_count = len(hdfs.stdout.readlines()) - 1
            thread_result = list()
            for part_item in range(part_count):
                data_cmd = ["hadoop", "fs", "-conf", f"{afs_conf}", "-D", "hadoop.job.ugi="f"{ugi}", "-cat", f"{data_hdfs}/part-{part_item}"]
                log.info(f"scan afs data cmd [{' '.join(data_cmd)}]")
                if len(thread_result) > self.info.get('max_records'): break
                hdfs = Popen(data_cmd, stdout=PIPE, stderr=DEVNULL)
                for i, line in enumerate(hdfs.stdout, 1):
                    if i > self.info.get('max_records'): break
                    thread_result.append(line.strip().decode())
            self.info['vec_dict'][data_key] = thread_result

    def build_annoy_new(self, vector_key):
        """annoy build"""
        import annoy
        if vector_key not in self.info: self.info[vector_key] = None
        self.info[vector_key] = annoy.AnnoyIndex(self.info["dim"], metric="angular")
        plan_type = self.info["plan_type"]
        vector_list = self.info["vec_dict"][vector_key]
        idx = 0
        for vec_item in vector_list:
            key, *vector = vec_item.strip().split()
            if vector_key not in self.info["keyIdx"]: self.info["keyIdx"][vector_key] = dict()
            if vector_key not in self.info["idxKey"]: self.info["idxKey"][vector_key] = dict()
            if len(vector) != self.info['dim'] or key in self.info["keyIdx"][vector_key]: continue
            if idx == 100: log.info(f"{key}")
            self.info["idxKey"][vector_key][idx] = key
            self.info["keyIdx"][vector_key][key] = idx
            self.info[vector_key].add_item(idx, [float(v) for v in vector])
            idx += 1
        log.info(f"{plan_type}-{vector_key}: annoy add item size: {len(self.info['idxKey'][vector_key])}")
        log.info(f"{plan_type}-{vector_key}: annoy start to build")
        self.info[vector_key].build(self.info['n_trees'])
        log.info(f"{plan_type}-{vector_key}: annoy build success")

    def create_dalton_client(self):
        """create dalton client"""
        from . import libdalton_cli
        libdalton_cli.close_log()
        self.info['cli'] = libdalton_cli.Client()
        conf_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], "conf")
        conf_file = f"{self.info['plugins_type']}-{self.info['data_type']}.conf"
        assert self.info['cli'].init(conf_dir, conf_file), "dalton client initialize failed."
        log.info(f"{self.info['data_type']}: create dalton client success.")

    def get_table_name(self):
        """get table"""
        if self.info['plugins_type'] == "shoubai" and self.info['data_type'] == "video":
            return "sv_duration"
        if self.info['plugins_type'] == "shoubai" and self.info['data_type'] == "news":
            return "news_readlist"

    def get_prefix(self):
        """get prefix"""
        if self.info['plugins_type'] == "shoubai":
            return "cl_news_" if self.info['data_type'] == "news" else "cl_pl_"

    def initialize(self):
        """initialize: download & annoy build"""
        self.input_data_ready()
        self.check_vector_info()
        self.filter_case_data()
        self.create_dalton_client()

    def haokan_sv_filter(self, beha):
        """filter haokan lianbo"""
        status = True
        if self.info['data_type'] == "video" and self.info['plan_type'] == "shoubai":
            log.info(f"duration:{beha.duration}; if play auto: {beha.auto_play}")
            status = False if beha.duration < 15 and beha.auto_play == True else True
        return status

    def load_truth_list(self, vector_type):
        """load truth list"""
        import lzo
        from . import feed_click_pb2
        from .libdalton_cli import sign_fs64, SNStatus; SUCC = SNStatus.SUCC
        log.info(f"{self.info['data_type']}: begin to get truth click list")
        table_name = self.get_table_name()
        user_feed_click = feed_click_pb2.UserFeedClick()
        if self.info['data_type'] == "news":
            check_ts = lambda beha: beha.ts < self.info['user_time']
        else:
            check_ts = lambda beha: beha.ts / 1000 < self.info['user_time']
        prefix = self.get_prefix()
        log.info(prefix)
        plan_type = self.info["plan_type"]
        case_data_list = self.info["case_data_list"]
        keys = [sign_fs64(prefix + key) for key in case_data_list]
        for i in range(0, len(case_data_list), 50):
            (status, result) = self.info['cli'].batch_get(table_name, keys[i:i+50])
            if status != SUCC: continue
            for (key, (sub_status, s)) in zip(case_data_list[i:i+50], result):
                if sub_status != SUCC: continue
                user_feed_click.ParseFromString(lzo.decompress(s, False, 128 * 1024 * 1024))
                truth_list = [str(beha.rid) for beha in user_feed_click.beha if check_ts(beha) and self.haokan_sv_filter(beha)]
                if len(truth_list) == 0: continue

                if vector_type not in self.info['truth']: self.info['truth'][vector_type] = dict()
                self.info['truth'][vector_type][key] = truth_list[-5:]
        log.info(f"{self.info['data_type']}-{vector_type}: There are {len(self.info['truth'][vector_type])} truth_list records")

    def load_recall_list(self, vector_type):
        """load recall list""" 
        import lzo
        from . import feed_click_pb2
        from .libdalton_cli import sign_fs64, SNStatus; SUCC = SNStatus.SUCC
        log.info(f"{self.info['data_type']}: begin to get recall click list")
        table_name = self.get_table_name()
        user_feed_click = feed_click_pb2.UserFeedClick()
        if self.info['data_type'] == "news":
            check_ts = lambda beha: beha.ts < self.info['user_time']
        else:
            check_ts = lambda beha: beha.ts / 1000 < self.info['user_time']
        prefix = self.get_prefix()
        sign = lambda recall_idx: sign_fs64(prefix + self.info['idxKey'][vector_type][recall_idx])
        case_data_list = self.info["case_data_list"]

        for key in case_data_list:
            if idx := self.info['keyIdx'][vector_type].get(key):
                recall_keys = [sign(i) for i in self.info[vector_type].get_nns_by_item(idx, self.info['search_n'])[1:]]
                (status, result) = self.info['cli'].batch_get(table_name, recall_keys)
                if SUCC != status: continue
                recall_dict = dict()
                for (sub_status, s) in result:
                    if sub_status != SUCC: continue
                    user_feed_click.ParseFromString(lzo.decompress(s, False, 128 * 1024 * 1024))
                    cnt = 0
                    for beha in user_feed_click.beha:
                        if check_ts(beha) and cnt < 20 and self.haokan_sv_filter(beha): 
                            recall_dict[str(beha.rid)] = recall_dict.get(str(beha.rid), 0) + 1
                            cnt += 1
                if len(recall_dict) == 0: continue
                recall_list = sorted(recall_dict.keys(),
                                     key=lambda k: recall_dict[k],
                                     reverse=True)
                if vector_type not in self.info['recall']: self.info['recall'][vector_type] = dict()
                self.info['recall'][vector_type][key] = recall_list[:1000]
        log.info(f"{self.info['data_type']}-{vector_type}: There are {len(self.info['recall'][vector_type])} recall_list records.")
    
    def cal_quasi_recall(self, vector_type):
        """calculate PR"""
        res = dict()
        log.info(f"{self.info['plan_type']}: start to cal quasi recall")
        truth_ratios, recall_ratios = list(), list()

        for (key, recall_list) in self.info['recall'].items():
            if truth_list := self.info['truth'].get(key):
                recall_truth_list = [i for i in truth_list if i in set(recall_list)]
                truth_ratios.append(len(recall_truth_list) / len(recall_list))
                recall_ratios.append(len(recall_truth_list) / len(truth_list))
        truth_ratio = sum(truth_ratios) / len(truth_ratios) if truth_ratios else 0
        recall_ratio = sum(recall_ratios) / len(recall_ratios) if recall_ratios else 0
        log.info(f"{self.info['plan_type']}-{vector_type}: truth_ratio={truth_ratio} recall_ratio={recall_ratio}")
        if "recall_ratio" not in res: res["recall_ratio"] = dict()
        if "truth_ratio" not in res: res["truth_ratio"] = dict()
        if vector_type not in res["recall_ratio"]: res["recall_ratio"][vector_type] = dict()
        if vector_type not in res["truth_ratio"]: res["truth_ratio"][vector_type] = dict()

        res["recall_ratio"][vector_type] = recall_ratio
        res["truth_ratio"][vector_type] = truth_ratio
        status = True
        return res, status
