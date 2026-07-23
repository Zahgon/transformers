
import argparse
import os
import tempfile
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import torch
from huggingface_hub import HfApi
from PIL import Image

from transformers import VJEPA2Config, VJEPA2Model, VJEPA2VideoProcessor
from transformers.models.vjepa2.modeling_vjepa2 import apply_masks


HUB_REPO = "https://github.com/facebookresearch/vjepa2"
HUB_SOURCE = "github"

HUB_MODELS = {
    "vit_large": "facebook/vjepa2-vitl-fpc64-256",
    "vit_huge": "facebook/vjepa2-vith-fpc64-256",
    "vit_giant": "facebook/vjepa2-vitg-fpc64-256",
    "vit_giant_384": "facebook/vjepa2-vitg-fpc64-384",
}

S3_MODELS = {
    "vit_large": "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
    "vit_huge": "https://dl.fbaipublicfiles.com/vjepa2/vith.pt",
    "vit_giant": "https://dl.fbaipublicfiles.com/vjepa2/vitg.pt",
    "vit_giant_384": "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt",
}

TOKEN = os.environ.get("HF_TOKEN", None)


def get_vjepa2_config(model_name):
    pass


def convert_encoder_keys(model_state_dict, og_encoder_state_dict, config):
    pass


def convert_predictor_keys(model_state_dict, og_predictor_state_dict, config):
    pass


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read())).convert("RGB")
    return image


def upload_original_ckpts(model_name):
    pass


@torch.no_grad()
def convert_and_test_vjepa2_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="vit_large",
        type=str,
        choices=[
            "vit_large",
            "vit_huge",
            "vit_giant",
            "vit_giant_384",
        ],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )
    parser.add_argument("--upload_original", action="store_true", help="upload the original checkpoint")

    args = parser.parse_args()
    convert_and_test_vjepa2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
    if args.upload_original:
        upload_original_ckpts(args.model_name)
