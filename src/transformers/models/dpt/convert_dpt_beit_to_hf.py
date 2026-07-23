
import argparse
from io import BytesIO
from pathlib import Path

import httpx
import torch
from PIL import Image

from transformers import BeitConfig, DPTConfig, DPTForDepthEstimation, DPTImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(model_name):
    pass


def create_rename_keys(config):
    rename_keys = []

    rename_keys.append(("pretrained.model.cls_token", "backbone.embeddings.cls_token"))
    rename_keys.append(("pretrained.model.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("pretrained.model.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))

    for i in range(config.backbone_config.num_hidden_layers):
        rename_keys.append((f"pretrained.model.blocks.{i}.gamma_1", f"backbone.encoder.layer.{i}.lambda_1"))
        rename_keys.append((f"pretrained.model.blocks.{i}.gamma_2", f"backbone.encoder.layer.{i}.lambda_2"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.output.dense.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.proj.weight", f"backbone.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.proj.bias", f"backbone.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.relative_position_bias_table", f"backbone.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table"))
        rename_keys.append((f"pretrained.model.blocks.{i}.attn.relative_position_index", f"backbone.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index"))

    for i in range(4):
        rename_keys.append((f"pretrained.act_postprocess{i+1}.0.project.0.weight", f"neck.reassemble_stage.readout_projects.{i}.0.weight"))
        rename_keys.append((f"pretrained.act_postprocess{i+1}.0.project.0.bias", f"neck.reassemble_stage.readout_projects.{i}.0.bias"))

        rename_keys.append((f"pretrained.act_postprocess{i+1}.3.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"pretrained.act_postprocess{i+1}.3.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"))

        if i != 2:
            rename_keys.append((f"pretrained.act_postprocess{i+1}.4.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"))
            rename_keys.append((f"pretrained.act_postprocess{i+1}.4.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"))

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


def read_in_q_k_v(state_dict, config):
    hidden_size = config.backbone_config.hidden_size
    for i in range(config.backbone_config.num_hidden_layers):
        in_proj_weight = state_dict.pop(f"pretrained.model.blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"pretrained.model.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"pretrained.model.blocks.{i}.attn.v_bias")
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = v_bias


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="dpt-beit-large-512",
        type=str,
        choices=["dpt-beit-large-512", "dpt-beit-large-384", "dpt-beit-base-384"],
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

    args = parser.parse_args()
    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
