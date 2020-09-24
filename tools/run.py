#!/usr/bin/env python
# coding=utf-8
"""
 @auth : wangna07@baidu.com
 @date : 2020-08-03 17:54:22
"""
import os
import shutil
import ujson
from quasi_config import config

if __name__ == "__main__":
    from app.main import MainEntry
    import logging; log = logging.getLogger()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recall", required=False)
    parser.add_argument("--truth", required=False)
    args = parser.parse_args()

    if not os.path.exists("log"): os.mkdir("log")
    logging.basicConfig(
        level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S",
        filename=f"log/{config.jobname}", filemode='w',
        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s")

    tmp_dir = os.path.join("tmp", config.jobname)
    args.tmp_dir = tmp_dir
    if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)
    elif os.path.exists(tmp_dir): 
        shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
    log.info(f"create tmp dir:{tmp_dir}")
    ctx = MainEntry(config, args)
    cal_res, status = ctx.cal_test()
    if status == False: 
        log.info(f"calculate task failed.")
        err_msg = "calculate task failed."
        result = dict(err_no=1, err_msg=err_msg)
    elif status == True:
        log.info(f"calcuate task success. result: {cal_res}")
        result = dict(err_no=0, truth_ratio=cal_res['truth_ratio'], recall_ratio=cal_res['recall_ratio'])
    print(ujson.dumps(result))


