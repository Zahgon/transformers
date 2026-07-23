
import argparse
import json
from io import BytesIO

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import BitImageProcessor, FocalNetConfig, FocalNetForImageClassification
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling


def get_focalnet_config(model_name):
    pass


def rename_key(name):
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    if "layers" in name:
        name = "encoder." + name
    if "encoder.layers" in name:
        name = name.replace("encoder.layers", "encoder.stages")
    if "downsample.proj" in name:
        name = name.replace("downsample.proj", "downsample.projection")
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    if "modulation.f.weight" in name or "modulation.f.bias" in name:
        name = name.replace("modulation.f", "modulation.projection_in")
    if "modulation.h.weight" in name or "modulation.h.bias" in name:
        name = name.replace("modulation.h", "modulation.projection_context")
    if "modulation.proj.weight" in name or "modulation.proj.bias" in name:
        name = name.replace("modulation.proj", "modulation.projection_out")

    if name == "norm.weight":
        name = "layernorm.weight"
    if name == "norm.bias":
        name = "layernorm.bias"

    if "head" in name:
        name = name.replace("head", "classifier")
    else:
        name = "focalnet." + name

    return name


def convert_focalnet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="focalnet-tiny",
        type=str,
        help="Name of the FocalNet model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub.",
    )

    args = parser.parse_args()
    convert_focalnet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
