#!/usr/bin/env bash
CONFIG=./projects/configs/maptr/maptr_nano_r18_110e.py
GPUS=1
CHECKPOINT=./ckpts/maptr_nano_r18_110e.pth
# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=7 \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval chamfer
