
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ConvNextConfig, ConvNextForImageClassification, ConvNextImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_convnext_config(checkpoint_url):
    pass


def rename_key(name):
    if "downsample_layers.0.0" in name:
        name = name.replace("downsample_layers.0.0", "embeddings.patch_embeddings")
    if "downsample_layers.0.1" in name:
        name = name.replace("downsample_layers.0.1", "embeddings.norm")  # we rename to layernorm later on
    if "downsample_layers.1.0" in name:
        name = name.replace("downsample_layers.1.0", "stages.1.downsampling_layer.0")
    if "downsample_layers.1.1" in name:
        name = name.replace("downsample_layers.1.1", "stages.1.downsampling_layer.1")
    if "downsample_layers.2.0" in name:
        name = name.replace("downsample_layers.2.0", "stages.2.downsampling_layer.0")
    if "downsample_layers.2.1" in name:
        name = name.replace("downsample_layers.2.1", "stages.2.downsampling_layer.1")
    if "downsample_layers.3.0" in name:
        name = name.replace("downsample_layers.3.0", "stages.3.downsampling_layer.0")
    if "downsample_layers.3.1" in name:
        name = name.replace("downsample_layers.3.1", "stages.3.downsampling_layer.1")
    if "stages" in name and "downsampling_layer" not in name:
        name = name[: len("stages.0")] + ".layers" + name[len("stages.0") :]
    if "stages" in name:
        name = name.replace("stages", "encoder.stages")
    if "norm" in name:
        name = name.replace("norm", "layernorm")
    if "gamma" in name:
        name = name.replace("gamma", "layer_scale_parameter")
    if "head" in name:
        name = name.replace("head", "classifier")

    return name


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_convnext_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
        type=str,
        help="URL of the original ConvNeXT checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory.",
    )

    args = parser.parse_args()
    convert_convnext_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
