
import argparse
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config, DPTImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(model_name):
    pass


def create_rename_keys(config):
    rename_keys = []

    rename_keys.append(("pretrained.cls_token", "backbone.embeddings.cls_token"))
    rename_keys.append(("pretrained.mask_token", "backbone.embeddings.mask_token"))
    rename_keys.append(("pretrained.pos_embed", "backbone.embeddings.position_embeddings"))
    rename_keys.append(("pretrained.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("pretrained.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))

    for i in range(config.backbone_config.num_hidden_layers):
        rename_keys.append((f"pretrained.blocks.{i}.ls1.gamma", f"backbone.encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"pretrained.blocks.{i}.ls2.gamma", f"backbone.encoder.layer.{i}.layer_scale2.lambda1"))
        rename_keys.append((f"pretrained.blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"pretrained.blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.norm2.bias"))
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.mlp.fc1.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.mlp.fc1.bias"))
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.mlp.fc2.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.mlp.fc2.bias"))
        rename_keys.append((f"pretrained.blocks.{i}.attn.proj.weight", f"backbone.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"pretrained.blocks.{i}.attn.proj.bias", f"backbone.encoder.layer.{i}.attention.output.dense.bias"))

    rename_keys.append(("pretrained.norm.weight", "backbone.layernorm.weight"))
    rename_keys.append(("pretrained.norm.bias", "backbone.layernorm.bias"))


    for i in range(4):
        rename_keys.append((f"depth_head.projects.{i}.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"depth_head.projects.{i}.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"))

        if i != 2:
            rename_keys.append((f"depth_head.resize_layers.{i}.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"))
            rename_keys.append((f"depth_head.resize_layers.{i}.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"))

    mapping = {1:3, 2:2, 3:1, 4:0}

    for i in range(1, 5):
        j = mapping[i]
        rename_keys.append((f"depth_head.scratch.refinenet{i}.out_conv.weight", f"neck.fusion_stage.layers.{j}.projection.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.out_conv.bias", f"neck.fusion_stage.layers.{j}.projection.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit1.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.bias"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.weight"))
        rename_keys.append((f"depth_head.scratch.refinenet{i}.resConfUnit2.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.bias"))

    for i in range(4):
        rename_keys.append((f"depth_head.scratch.layer{i+1}_rn.weight", f"neck.convs.{i}.weight"))

    rename_keys.append(("depth_head.scratch.output_conv1.weight", "head.conv1.weight"))
    rename_keys.append(("depth_head.scratch.output_conv1.bias", "head.conv1.bias"))
    rename_keys.append(("depth_head.scratch.output_conv2.0.weight", "head.conv2.weight"))
    rename_keys.append(("depth_head.scratch.output_conv2.0.bias", "head.conv2.bias"))
    rename_keys.append(("depth_head.scratch.output_conv2.2.weight", "head.conv3.weight"))
    rename_keys.append(("depth_head.scratch.output_conv2.2.bias", "head.conv3.bias"))

    return rename_keys


def read_in_q_k_v(state_dict, config):
    hidden_size = config.backbone_config.hidden_size
    for i in range(config.backbone_config.num_hidden_layers):
        in_proj_weight = state_dict.pop(f"pretrained.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"pretrained.blocks.{i}.attn.qkv.bias")
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
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


name_to_checkpoint = {
    "depth-anything-small": "pytorch_model.bin",
    "depth-anything-base": "pytorch_model.bin",
    "depth-anything-large": "pytorch_model.bin",
    "depth-anything-v2-small": "depth_anything_v2_vits.pth",
    "depth-anything-v2-base": "depth_anything_v2_vitb.pth",
    "depth-anything-v2-large": "depth_anything_v2_vitl.pth",
    "depth-anything-v2-metric-indoor-small": "depth_anything_v2_metric_hypersim_vits.pth",
    "depth-anything-v2-metric-indoor-base": "depth_anything_v2_metric_hypersim_vitb.pth",
    "depth-anything-v2-metric-indoor-large": "depth_anything_v2_metric_hypersim_vitl.pth",
    "depth-anything-v2-metric-outdoor-small": "depth_anything_v2_metric_vkitti_vits.pth",
    "depth-anything-v2-metric-outdoor-base": "depth_anything_v2_metric_vkitti_vitb.pth",
    "depth-anything-v2-metric-outdoor-large": "depth_anything_v2_metric_vkitti_vitl.pth",
}


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, verify_logits):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="depth-anything-small",
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
