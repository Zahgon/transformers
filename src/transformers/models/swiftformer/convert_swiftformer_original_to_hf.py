
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    SwiftFormerConfig,
    SwiftFormerForImageClassification,
    ViTImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

device = torch.device("cpu")


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


def get_expected_output(swiftformer_name):
    pass


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def create_rename_keys(state_dict):
    rename_keys = []
    for k in state_dict:
        k_new = k
        if ".pwconv" in k:
            k_new = k_new.replace(".pwconv", ".point_wise_conv")
        if ".dwconv" in k:
            k_new = k_new.replace(".dwconv", ".depth_wise_conv")
        if ".Proj." in k:
            k_new = k_new.replace(".Proj.", ".proj.")
        if "patch_embed" in k_new:
            k_new = k_new.replace("patch_embed", "swiftformer.patch_embed.patch_embedding")
        if "network" in k_new:
            ls = k_new.split(".")
            if ls[2].isdigit():
                k_new = "swiftformer.encoder.network." + ls[1] + ".blocks." + ls[2] + "." + ".".join(ls[3:])
            else:
                k_new = k_new.replace("network", "swiftformer.encoder.network")
        rename_keys.append((k, k_new))
    return rename_keys


@torch.no_grad()
def convert_swiftformer_checkpoint(swiftformer_name, pytorch_dump_folder_path, original_ckpt):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--swiftformer_name",
        default="swiftformer_xs",
        choices=["swiftformer_xs", "swiftformer_s", "swiftformer_l1", "swiftformer_l3"],
        type=str,
        help="Name of the SwiftFormer model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="./converted_outputs/",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--original_ckpt", default=None, type=str, help="Path to the original model checkpoint.")

    args = parser.parse_args()
    convert_swiftformer_checkpoint(args.swiftformer_name, args.pytorch_dump_folder_path, args.original_ckpt)
