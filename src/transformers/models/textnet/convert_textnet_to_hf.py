
import argparse
import json
import logging
import re
from collections import OrderedDict
from io import BytesIO

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import TextNetBackbone, TextNetConfig, TextNetImageProcessor


tiny_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_tiny.config"
small_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_small.config"
base_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_base.config"

rename_key_mappings = {
    "module.backbone": "textnet",
    "first_conv": "stem",
    "bn": "batch_norm",
    "ver": "vertical",
    "hor": "horizontal",
}


def prepare_config(size_config_url, size):
    pass


def convert_textnet_checkpoint(checkpoint_url, checkpoint_config_filename, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/czczup/FAST/releases/download/release/fast_base_ic17mlt_640.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--checkpoint_config_filename",
        default="fast_base_ic17mlt_640.py",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()

    convert_textnet_checkpoint(
        args.checkpoint_url,
        args.checkpoint_config_filename,
        args.pytorch_dump_folder_path,
    )
