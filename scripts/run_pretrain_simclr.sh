#!/bin/bash

# SimCLR 預訓練腳本
# 使用方式: bash scripts/run_pretrain_simclr.sh [config_name]
# 預設使用 simclr config

CONFIG=${1:-simclr}
uv run pretrain_simclr.py --config "$CONFIG"

