
import argparse
import json
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import PoolFormerConfig, PoolFormerForImageClassification, PoolFormerImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def replace_key_with_offset(key, offset, original_name, new_name):
    """
    Replaces the key by subtracting the offset from the original layer number
    """
    to_find = original_name.split(".")[0]
    key_list = key.split(".")
    orig_block_num = int(key_list[key_list.index(to_find) - 2])
    layer_num = int(key_list[key_list.index(to_find) - 1])
    new_block_num = orig_block_num - offset

    key = key.replace(f"{orig_block_num}.{layer_num}.{original_name}", f"block.{new_block_num}.{layer_num}.{new_name}")
    return key


def rename_keys(state_dict):
    new_state_dict = OrderedDict()
    total_embed_found, patch_emb_offset = 0, 0
    for key, value in state_dict.items():
        if key.startswith("network"):
            key = key.replace("network", "poolformer.encoder")
        if "proj" in key:
            if key.endswith("bias") and "patch_embed" not in key:
                patch_emb_offset += 1
            to_replace = key[: key.find("proj")]
            key = key.replace(to_replace, f"patch_embeddings.{total_embed_found}.")
            key = key.replace("proj", "projection")
            if key.endswith("bias"):
                total_embed_found += 1
        if "patch_embeddings" in key:
            key = "poolformer.encoder." + key
        if "mlp.fc1" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "mlp.fc1", "output.conv1")
        if "mlp.fc2" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "mlp.fc2", "output.conv2")
        if "norm1" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "norm1", "before_norm")
        if "norm2" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "norm2", "after_norm")
        if "layer_scale_1" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "layer_scale_1", "layer_scale_1")
        if "layer_scale_2" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "layer_scale_2", "layer_scale_2")
        if "head" in key:
            key = key.replace("head", "classifier")
        new_state_dict[key] = value
    return new_state_dict


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_poolformer_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="poolformer_s12",
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
    convert_poolformer_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)
