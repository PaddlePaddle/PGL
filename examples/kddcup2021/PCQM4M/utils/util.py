#-*- coding: utf-8 -*-
import os
import sys
import warnings


def get_last_dir(path):
    """Get the last directory of a path.
    """
    if os.path.isfile(path):
        # e.g: "../checkpoints/task_name/epoch0_step300/predict.txt"
        # return "epoch0_step300"
        last_dir = path.split("/")[-2]

    elif os.path.isdir(path):
        if path[-1] == '/':
            # e.g: "../checkpoints/task_name/epoch0_step300/"
            last_dir = path.split('/')[-2]
        else:
            # e.g: "../checkpoints/task_name/epoch0_step300"
            last_dir = path.split('/')[-1]
    else:
        # path or file is not existed
        warnings.warn('%s is not a existed file or path' % path)
        last_dir = ""

    return last_dir

def make_dir(path):
    """Build directory"""
    if not os.path.exists(path):
        os.makedirs(path)
