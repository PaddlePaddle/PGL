#!/bin/bash

TAG_NAME=pglbox:cuda11.0-cudnn8-v1.0
docker build --network host -t ${TAG_NAME} -f ./Dockerfile . 
