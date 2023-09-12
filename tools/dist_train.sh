#!/usr/bin/env bash

# CONFIG=$1
# GPUS=$2
# WORK_DIRS=$3
CONFIG=./projects/configs/maptr/maptr_nano_r18_110e.py
GPUS=6
WORK_DIRS=./output/maptr_nano_r18_110e
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --work-dir $WORK_DIRS --launcher pytorch ${@:3} --deterministic
