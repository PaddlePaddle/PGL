#!/bin/bash


BASE_BUILD_IMAGE=registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.0-cudnn8
TAG_NAME=pglbox:cuda11.0-cudnn8-v1.0

docker build --network host -t ${TAG_NAME} -f ./Dockerfile . 
