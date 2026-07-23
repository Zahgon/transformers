
import argparse
import json
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    SegformerConfig,
    SegformerForImageClassification,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_keys(state_dict, encoder_only=False):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if encoder_only and not key.startswith("head"):
            key = "segformer.encoder." + key
        if key.startswith("backbone"):
            key = key.replace("backbone", "segformer.encoder")
        if "patch_embed" in key:
            idx = key[key.find("patch_embed") + len("patch_embed")]
            key = key.replace(f"patch_embed{idx}", f"patch_embeddings.{int(idx) - 1}")
        if "norm" in key:
            key = key.replace("norm", "layer_norm")
        if "segformer.encoder.layer_norm" in key:
            idx = key[key.find("segformer.encoder.layer_norm") + len("segformer.encoder.layer_norm")]
            key = key.replace(f"layer_norm{idx}", f"layer_norm.{int(idx) - 1}")
        if "layer_norm1" in key:
            key = key.replace("layer_norm1", "layer_norm_1")
        if "layer_norm2" in key:
            key = key.replace("layer_norm2", "layer_norm_2")
        if "block" in key:
            idx = key[key.find("block") + len("block")]
            key = key.replace(f"block{idx}", f"block.{int(idx) - 1}")
        if "attn.q" in key:
            key = key.replace("attn.q", "attention.self.query")
        if "attn.proj" in key:
            key = key.replace("attn.proj", "attention.output.dense")
        if "attn" in key:
            key = key.replace("attn", "attention.self")
        if "fc1" in key:
            key = key.replace("fc1", "dense1")
        if "fc2" in key:
            key = key.replace("fc2", "dense2")
        if "linear_pred" in key:
            key = key.replace("linear_pred", "classifier")
        if "linear_fuse" in key:
            key = key.replace("linear_fuse.conv", "linear_fuse")
            key = key.replace("linear_fuse.bn", "batch_norm")
        if "linear_c" in key:
            idx = key[key.find("linear_c") + len("linear_c")]
            key = key.replace(f"linear_c{idx}", f"linear_c.{int(idx) - 1}")
        if key.startswith("head"):
            key = key.replace("head", "classifier")
        new_state_dict[key] = value

    return new_state_dict


def read_in_k_v(state_dict, config):
    pass


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_segformer_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="segformer.b0.512x512.ade.160k",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original PyTorch checkpoint (.pth file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_segformer_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)
