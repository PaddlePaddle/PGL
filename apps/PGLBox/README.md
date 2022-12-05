# PGLBox: Distributed Hierarchical GPU-based Engine for Efficiently Training Super-large Scale Graph Neural Network

## 快速启动

```
git clone https://github.com/PaddlePaddle/PGL

cd PGL/apps/PGLBox

# 如果自行构建docker
cd docker/
sh build.sh
cd ../

# 按照教程放置 数据和配置到本目录

IMAGE_NAME = 

nvidia-docker run -it --rm \
    --name ${name} \
    --network host \
    --ipc=host \
    -v ${PWD}:/pglbox \
    ${IMAGE_NAME} /pglbox/train.sh

```
