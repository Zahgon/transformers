
import argparse
import json
import re
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    MobileNetV1Config,
    MobileNetV1ForImageClassification,
    MobileNetV1ImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def _build_tf_to_pytorch_map(model, config, tf_weights=None):
    pass


def load_tf_weights_in_mobilenet_v1(model, config, tf_checkpoint_path):
    pass


def get_mobilenet_v1_config(model_name):
    pass


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_movilevit_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="mobilenet_v1_1.0_224",
        type=str,
        help="Name of the MobileNetV1 model you'd like to convert. Should in the form 'mobilenet_v1_<depth>_<size>'.",
    )
    parser.add_argument(
        "--checkpoint_path", required=True, type=str, help="Path to the original TensorFlow checkpoint (.ckpt file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_movilevit_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
