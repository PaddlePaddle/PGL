#-*- coding: utf-8 -*-
import os
import sys
import six
import yaml
import glob
import shutil
import random
import logging
import time
import datetime
import warnings
import numpy as np

def make_config_file(template_config_file):
    cfg = yaml.load(open(template_config_file), Loader=yaml.FullLoader)

    filename = "outputs"

    for cv in range(5):
        cfg_name = "configs/r_unimp_peg_gpr_%s.yaml" % cv
        if os.path.exists(cfg_name):
            os.remove(cfg_name)
        
        cfg["valid_name"] = "valid_%s.npy" % cv
        cfg["test_name"] = "test_%s" % cv
        cfg["model_output_path"] = "./%s/model/cv_%s" % (filename, cv)
        cfg["model_result_path"] = "./%s/result/cv_%s" % (filename, cv)
        with open(cfg_name, 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)


if __name__=="__main__":
    template_config_file = sys.argv[1]
    make_config_file(template_config_file)
