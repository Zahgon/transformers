
import argparse
from io import BytesIO
from pathlib import Path

import httpx
import torch
from PIL import Image

from transformers import DPTConfig, DPTForDepthEstimation, DPTImageProcessor, Swinv2Config
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(model_name):
    pass


def create_rename_keys(config):
    rename_keys = []

    rename_keys.append(("pretrained.model.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("pretrained.model.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("pretrained.model.patch_embed.norm.weight", "backbone.embeddings.norm.weight"))
    rename_keys.append(("pretrained.model.patch_embed.norm.bias", "backbone.embeddings.norm.bias"))

    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.logit_scale", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.logit_scale"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.cpb_mlp.0.weight", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.continuous_position_bias_mlp.0.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.cpb_mlp.0.bias", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.continuous_position_bias_mlp.0.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.cpb_mlp.2.weight", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.continuous_position_bias_mlp.2.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.q_bias", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.v_bias", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.proj.weight", f"backbone.encoder.layers.{i}.blocks.{j}.attention.output.dense.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.attn.proj.bias", f"backbone.encoder.layers.{i}.blocks.{j}.attention.output.dense.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.norm1.weight", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_before.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.norm1.bias", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_before.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.mlp.fc1.weight", f"backbone.encoder.layers.{i}.blocks.{j}.intermediate.dense.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.mlp.fc1.bias", f"backbone.encoder.layers.{i}.blocks.{j}.intermediate.dense.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.mlp.fc2.weight", f"backbone.encoder.layers.{i}.blocks.{j}.output.dense.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.mlp.fc2.bias", f"backbone.encoder.layers.{i}.blocks.{j}.output.dense.bias"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.norm2.weight", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_after.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.blocks.{j}.norm2.bias", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_after.bias"))

        if i in [0,1,2]:
            rename_keys.append((f"pretrained.model.layers.{i}.downsample.reduction.weight", f"backbone.encoder.layers.{i}.downsample.reduction.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.downsample.norm.weight", f"backbone.encoder.layers.{i}.downsample.norm.weight"))
            rename_keys.append((f"pretrained.model.layers.{i}.downsample.norm.bias", f"backbone.encoder.layers.{i}.downsample.norm.bias"))


    mapping = {1:3, 2:2, 3:1, 4:0}

    for i in range(1, 5):
        j = mapping[i]
        rename_keys.append((f"scratch.refinenet{i}.out_conv.weight", f"neck.fusion_stage.layers.{j}.projection.weight"))
        rename_keys.append((f"scratch.refinenet{i}.out_conv.bias", f"neck.fusion_stage.layers.{j}.projection.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.bias"))

    for i in range(4):
        rename_keys.append((f"scratch.layer{i+1}_rn.weight", f"neck.convs.{i}.weight"))

    for i in range(0, 5, 2):
        rename_keys.append((f"scratch.output_conv.{i}.weight", f"head.head.{i}.weight"))
        rename_keys.append((f"scratch.output_conv.{i}.bias", f"head.head.{i}.bias"))

    return rename_keys


def remove_ignore_keys_(state_dict):
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def read_in_q_k_v(state_dict, config, model):
    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            dim = model.backbone.encoder.layers[i].blocks[j].attention.self.all_head_size
            in_proj_weight = state_dict.pop(f"pretrained.model.layers.{i}.blocks.{j}.attn.qkv.weight")
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[
                dim : dim * 2, :
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[
                -dim:, :
            ]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, verify_logits, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="dpt-swinv2-base-384",
        type=str,
        choices=["dpt-swinv2-tiny-256", "dpt-swinv2-base-384", "dpt-swinv2-large-384"],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--verify_logits",
        action="store_true",
        help="Whether to verify logits after conversion.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub after conversion.",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.verify_logits, args.push_to_hub)
