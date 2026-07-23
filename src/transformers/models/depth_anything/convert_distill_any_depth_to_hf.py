
import argparse
import re
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config, DPTImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"(backbone|pretrained)\.cls_token": r"backbone.embeddings.cls_token",
    r"(backbone|pretrained)\.mask_token": r"backbone.embeddings.mask_token",
    r"(backbone|pretrained)\.pos_embed": r"backbone.embeddings.position_embeddings",
    r"(backbone|pretrained)\.patch_embed\.proj\.(weight|bias)": r"backbone.embeddings.patch_embeddings.projection.\2",
    r"(backbone|pretrained)\.norm\.(weight|bias)": r"backbone.layernorm.\2",
    r"(backbone|pretrained)(\.blocks(\.\d+)?)?\.(\d+)\.attn\.proj\.(weight|bias)": r"backbone.encoder.layer.\4.attention.output.dense.\5",
    r"(backbone|pretrained)(\.blocks(\.\d+)?)?\.(\d+)\.ls(1|2)\.gamma": r"backbone.encoder.layer.\4.layer_scale\5.lambda1",
    r"(backbone|pretrained)(\.blocks(\.\d+)?)?\.(\d+)\.mlp\.fc(1|2)\.(weight|bias)": r"backbone.encoder.layer.\4.mlp.fc\5.\6",
    r"(backbone|pretrained)(\.blocks(\.\d+)?)?\.(\d+)\.norm(1|2)\.(weight|bias)": r"backbone.encoder.layer.\4.norm\5.\6",
    r"depth_head\.projects\.(\d+)\.(weight|bias)": r"neck.reassemble_stage.layers.\1.projection.\2",
    r"depth_head\.resize_layers\.(?!2)(\d+)\.(weight|bias)": r"neck.reassemble_stage.layers.\1.resize.\2",
    r"depth_head\.scratch\.layer(\d+)_rn\.weight": lambda m: f"neck.convs.{int(m[1]) - 1}.weight",
    r"depth_head\.scratch\.output_conv(\d+)(?:\.(\d+))?\.(weight|bias)": lambda m: (
        f"head.conv{int(m[1]) + (int(m[2]) // 2 if m[2] else 0)}.{m[3]}" if m[1] == "2" else f"head.conv{m[1]}.{m[3]}"
    ),
    r"depth_head\.scratch\.refinenet(\d+)\.out_conv\.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{3 - (int(m[1]) - 1)}.projection.{m[2]}",
    r"depth_head\.scratch\.refinenet(\d+)\.resConfUnit(\d+)\.conv(\d+)\.(weight|bias)": lambda m: f"neck.fusion_stage.layers.{3 - (int(m[1]) - 1)}.residual_layer{m[2]}.convolution{m[3]}.{m[4]}",
}


def get_dpt_config(model_name):
    pass


def convert_key_pattern(key, mapping):
    pass


def convert_keys(state_dict, config):
    pass


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


name_to_checkpoint = {
    "distill-any-depth-small": "small/model.safetensors",
    "distill-any-depth-base": "base/model.safetensors",
    "distill-any-depth-large": "large/model.safetensors",
}


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="distill-any-depth-small",
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
        action="store_true",
        required=False,
        help="Whether to verify the logits after conversion.",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
