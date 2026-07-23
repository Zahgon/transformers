
import argparse
import json
import os
from io import BytesIO

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ConvNextImageProcessor, ConvNextV2Config, ConvNextV2ForImageClassification
from transformers.image_utils import PILImageResampling
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_convnextv2_config(checkpoint_url):
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
    if "gamma" in name:
        name = name.replace("gamma", "weight")
    if "beta" in name:
        name = name.replace("beta", "bias")
    if "stages" in name:
        name = name.replace("stages", "encoder.stages")
    if "norm" in name:
        name = name.replace("norm", "layernorm")
    if "head" in name:
        name = name.replace("head", "classifier")

    return name


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


def convert_preprocessor(checkpoint_url):
    pass


@torch.no_grad()
def convert_convnextv2_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt",
        type=str,
        help="URL of the original ConvNeXTV2 checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image preprocessor to the hub")

    args = parser.parse_args()
    convert_convnextv2_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub
    )
