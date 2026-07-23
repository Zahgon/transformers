
import argparse
import re
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    Dinov2Config,
    PromptDepthAnythingConfig,
    PromptDepthAnythingForDepthEstimation,
    PromptDepthAnythingImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(model_name):
    pass


def transform_qkv_weights(key, value, config):
    pass


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"pretrained.cls_token": r"backbone.embeddings.cls_token",
    r"pretrained.mask_token": r"backbone.embeddings.mask_token",
    r"pretrained.pos_embed": r"backbone.embeddings.position_embeddings",
    r"pretrained.patch_embed.proj.(weight|bias)": r"backbone.embeddings.patch_embeddings.projection.\1",
    r"pretrained.norm.(weight|bias)": r"backbone.layernorm.\1",
    r"pretrained.blocks.(\d+).ls1.gamma": r"backbone.encoder.layer.\1.layer_scale1.lambda1",
    r"pretrained.blocks.(\d+).ls2.gamma": r"backbone.encoder.layer.\1.layer_scale2.lambda1",
    r"pretrained.blocks.(\d+).norm1.(weight|bias)": r"backbone.encoder.layer.\1.norm1.\2",
    r"pretrained.blocks.(\d+).norm2.(weight|bias)": r"backbone.encoder.layer.\1.norm2.\2",
    r"pretrained.blocks.(\d+).mlp.fc1.(weight|bias)": r"backbone.encoder.layer.\1.mlp.fc1.\2",
    r"pretrained.blocks.(\d+).mlp.fc2.(weight|bias)": r"backbone.encoder.layer.\1.mlp.fc2.\2",
    r"pretrained.blocks.(\d+).attn.proj.(weight|bias)": r"backbone.encoder.layer.\1.attention.output.dense.\2",
    r"pretrained.blocks.(\d+).attn.qkv.(weight|bias)": r"qkv_transform_\2_\1",
    r"depth_head.projects.(\d+).(weight|bias)": r"neck.reassemble_stage.layers.\1.projection.\2",
    r"depth_head.scratch.layer(\d+)_rn.weight": lambda m: f"neck.convs.{int(m.group(1)) - 1}.weight",
    r"depth_head.resize_layers.(\d+).(weight|bias)": r"neck.reassemble_stage.layers.\1.resize.\2",
    r"depth_head.scratch.refinenet(\d+).out_conv.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{4 - int(m.group(1))}.projection.{m.group(2)}",
    r"depth_head.scratch.refinenet(\d+).resConfUnit1.conv1.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{4 - int(m.group(1))}.residual_layer1.convolution1.{m.group(2)}",
    r"depth_head.scratch.refinenet(\d+).resConfUnit1.conv2.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{4 - int(m.group(1))}.residual_layer1.convolution2.{m.group(2)}",
    r"depth_head.scratch.refinenet(\d+).resConfUnit2.conv1.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{4 - int(m.group(1))}.residual_layer2.convolution1.{m.group(2)}",
    r"depth_head.scratch.refinenet(\d+).resConfUnit2.conv2.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{4 - int(m.group(1))}.residual_layer2.convolution2.{m.group(2)}",
    r"depth_head.scratch.refinenet(\d+).resConfUnit_depth.0.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{4 - int(m.group(1))}.prompt_depth_layer.convolution1.{m.group(2)}",
    r"depth_head.scratch.refinenet(\d+).resConfUnit_depth.2.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{4 - int(m.group(1))}.prompt_depth_layer.convolution2.{m.group(2)}",
    r"depth_head.scratch.refinenet(\d+).resConfUnit_depth.4.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{4 - int(m.group(1))}.prompt_depth_layer.convolution3.{m.group(2)}",
    r"depth_head.scratch.output_conv1.(weight|bias)": r"head.conv1.\1",
    r"depth_head.scratch.output_conv2.0.(weight|bias)": r"head.conv2.\1",
    r"depth_head.scratch.output_conv2.2.(weight|bias)": r"head.conv3.\1",
}


def convert_old_keys_to_new_keys(state_dict_keys: dict | None = None):
    """
    Convert old state dict keys to new keys using regex patterns.
    """
    output_dict = {}
    if state_dict_keys is not None:
        for old_key in state_dict_keys:
            new_key = old_key
            for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
                match = re.match(pattern, old_key)
                if match:
                    if callable(replacement):
                        new_key = replacement(match)
                    else:
                        new_key = re.sub(pattern, replacement, old_key)
                    break
            output_dict[old_key] = new_key
    return output_dict


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    pass


name_to_checkpoint = {
    "prompt-depth-anything-vits": "model.ckpt",
    "prompt-depth-anything-vits-transparent": "model.ckpt",
    "prompt-depth-anything-vitl": "model.ckpt",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="prompt_depth_anything_vits",
        type=str,
        choices=name_to_checkpoint.keys(),
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
        help="Whether to push the model to the hub after conversion.",
    )
    parser.add_argument(
        "--verify_logits",
        action="store_false",
        required=False,
        help="Whether to verify the logits after conversion.",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
