#!/bin/bash

# 此腳本用於分割任務訓練（UNet, TransUNet）
# SimCLR 預訓練請使用: uv run pretrain_simclr.py --config simclr

CONFIG=${1:-transunet}
uv run train.py --config "$CONFIG"
