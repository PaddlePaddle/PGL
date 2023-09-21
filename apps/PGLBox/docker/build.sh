#!/bin/bash

TAG_NAME=pgl:pglbox-0.0-cuda11.0-cudnn8
docker build --network host -t ${TAG_NAME} -f ./Dockerfile . 
