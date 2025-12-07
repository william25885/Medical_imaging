#!/bin/bash

# TODO: Download your checkpoint by gdown and unzip it here if needed

# Run the submission generation script for Problem 2: TransUNet (without pretrained encoder)
uv run generate_submission.py --config transunet_no_pretrain --public_dir "${1}" --private_dir "${2}"

