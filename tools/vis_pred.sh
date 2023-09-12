#!/usr/bin/env bash

CONFIG=./projects/configs/maptr/maptr_nano_r18_110e.py
CHECKPOINT=./ckpts/maptr_nano_r18_110e.pth

# CONFIG=./projects/configs/maptr/maptr_tiny_r50_110e.py
# CHECKPOINT=./ckpts/maptr_tiny_r50_110e.pth


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 \
python tools/maptr/vis_pred.py $CONFIG $CHECKPOINT \
                    --score-thresh 0.2 \
                    --gt-format fixed_num_pts \
                    --show-dir ./viz/nano
