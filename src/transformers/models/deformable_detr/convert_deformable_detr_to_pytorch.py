
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_key(orig_key):
    if "backbone.0.body" in orig_key:
        orig_key = orig_key.replace("backbone.0.body", "backbone.conv_encoder.model")
    if "transformer" in orig_key:
        orig_key = orig_key.replace("transformer.", "")
    if "norm1" in orig_key:
        if "encoder" in orig_key:
            orig_key = orig_key.replace("norm1", "self_attn_layer_norm")
        else:
            orig_key = orig_key.replace("norm1", "encoder_attn_layer_norm")
    if "norm2" in orig_key:
        if "encoder" in orig_key:
            orig_key = orig_key.replace("norm2", "final_layer_norm")
        else:
            orig_key = orig_key.replace("norm2", "self_attn_layer_norm")
    if "norm3" in orig_key:
        orig_key = orig_key.replace("norm3", "final_layer_norm")
    if "linear1" in orig_key:
        orig_key = orig_key.replace("linear1", "fc1")
    if "linear2" in orig_key:
        orig_key = orig_key.replace("linear2", "fc2")
    if "query_embed" in orig_key:
        orig_key = orig_key.replace("query_embed", "query_position_embeddings")
    if "cross_attn" in orig_key:
        orig_key = orig_key.replace("cross_attn", "encoder_attn")

    return orig_key


def read_in_q_k_v(state_dict):
    for i in range(6):
        in_proj_weight = state_dict.pop(f"decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"decoder.layers.{i}.self_attn.in_proj_bias")
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))

    return image


@torch.no_grad()
def convert_deformable_detr_checkpoint(
    checkpoint_path,
    single_scale,
    dilation,
    with_box_refine,
    two_stage,
    pytorch_dump_folder_path,
    push_to_hub,
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/niels/checkpoints/deformable_detr/r50_deformable_detr-checkpoint.pth",
        help="Path to Pytorch checkpoint (.pth file) you'd like to convert.",
    )
    parser.add_argument("--single_scale", action="store_true", help="Whether to set config.num_features_levels = 1.")
    parser.add_argument("--dilation", action="store_true", help="Whether to set config.dilation=True.")
    parser.add_argument("--with_box_refine", action="store_true", help="Whether to set config.with_box_refine=True.")
    parser.add_argument("--two_stage", action="store_true", help="Whether to set config.two_stage=True.")
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to output PyTorch model.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )
    args = parser.parse_args()
    convert_deformable_detr_checkpoint(
        args.checkpoint_path,
        args.single_scale,
        args.dilation,
        args.with_box_refine,
        args.two_stage,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
