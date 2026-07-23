
import argparse
import itertools
import math
from io import BytesIO
from pathlib import Path

import httpx
import torch
from PIL import Image
from torchvision import transforms

from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(model_name):
    pass


def create_rename_keys_dpt(config):
    pass


def create_rename_keys_backbone(config):
    pass


def read_in_q_k_v(state_dict, config):
    for i in range(config.backbone_config.num_hidden_layers):
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        hidden_size = config.backbone_config.hidden_size
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            hidden_size : hidden_size * 2
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-hidden_size:]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def prepare_img():
    url = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


name_to_url = {
    "dpt-dinov2-small-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_dpt_head.pth",
    "dpt-dinov2-small-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_kitti_dpt_head.pth",
    "dpt-dinov2-base-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_nyu_dpt_head.pth",
    "dpt-dinov2-base-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_kitti_dpt_head.pth",
    "dpt-dinov2-large-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_nyu_dpt_head.pth",
    "dpt-dinov2-large-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_kitti_dpt_head.pth",
    "dpt-dinov2-giant-nyu": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_nyu_dpt_head.pth",
    "dpt-dinov2-giant-kitti": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_kitti_dpt_head.pth",
}


def get_original_pixel_values(image):
    pass


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="dpt-dinov2-small-nyu",
        type=str,
        choices=name_to_url.keys(),
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
        help="Path to the output PyTorch model directory.",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits)
