#-*- coding: utf-8 -*-
import sys
import os
import logging

def prepare_logger(log_dir=None, log_filename="log.txt", stdout=False):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
            fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s')

    if stdout or log_dir is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        #  handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    if log_dir is not None:
        if os.path.isdir(log_dir):
            handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        else:
            handler = logging.FileHandler(log_dir)
        handler.setFormatter(formatter)
        #  handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    logger.propagate = False

    return logger

def log_to_file(logger, log_dir, log_filename="log.txt"):
    if os.path.isdir(log_dir):
        handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    else:
        handler = logging.FileHandler(log_dir)

    formatter = logging.Formatter(
            fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)


