#!/usr/bin/env bash

CONFIG=./projects/configs/maptr/maptr_nano_r18_110e.py
CHECKPOINT=./ckpts/maptr_nano_r18_110e.pth


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
python tools/maptr/benchmark.py $CONFIG \
                    --checkpoint $CHECKPOINT 
