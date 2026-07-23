
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from transformers import BitConfig, BitForImageClassification, BitImageProcessor
from transformers.image_utils import PILImageResampling
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_config(model_name):
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    conv_layer = "std_conv" if "bit" in model_name else False

    config = BitConfig(
        conv_layer=conv_layer,
        num_labels=1000,
        id2label=id2label,
        label2id=label2id,
    )

    return config


def rename_key(name):
    if "stem.conv" in name:
        name = name.replace("stem.conv", "bit.embedder.convolution")
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    if "head.fc" in name:
        name = name.replace("head.fc", "classifier.1")
    if name.startswith("norm"):
        name = "bit." + name
    if "bit" not in name and "classifier" not in name:
        name = "bit.encoder." + name

    return name


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_bit_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="resnetv2_50x1_bitm",
        type=str,
        help="Name of the BiT timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub.",
    )

    args = parser.parse_args()
    convert_bit_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
