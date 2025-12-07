#!/bin/bash

# TODO: Download your checkpoint by gdown and unzip it here if needed

# Run the submission generation script for TransUNet with SimCLR pretrained encoder
uv run generate_submission.py --config transunet --public_dir "${1}" --private_dir "${2}"
