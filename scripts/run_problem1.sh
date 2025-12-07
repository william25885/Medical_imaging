#!/bin/bash

# TODO: Download your checkpoint by gdown and unzip it here if needed

# Run the submission generation script for Problem 1: Simple-CNN UNet
uv run generate_submission.py --config cnn_unet --public_dir "${1}" --private_dir "${2}"

