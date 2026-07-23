
import argparse
import json
from io import BytesIO

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import SegformerImageProcessor, SwinConfig, UperNetConfig, UperNetForSemanticSegmentation


def get_upernet_config(model_name):
    pass


def create_rename_keys(config):
    rename_keys = []

    rename_keys.append(("backbone.patch_embed.projection.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.projection.bias", "backbone.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.patch_embed.norm.weight", "backbone.embeddings.norm.weight"))
    rename_keys.append(("backbone.patch_embed.norm.bias", "backbone.embeddings.norm.bias"))
    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm1.weight", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_before.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm1.bias", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_before.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.relative_position_bias_table", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_bias_table"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.relative_position_index", f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_index"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.proj.weight", f"backbone.encoder.layers.{i}.blocks.{j}.attention.output.dense.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.attn.w_msa.proj.bias", f"backbone.encoder.layers.{i}.blocks.{j}.attention.output.dense.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm2.weight", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_after.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.norm2.bias", f"backbone.encoder.layers.{i}.blocks.{j}.layernorm_after.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.0.0.weight", f"backbone.encoder.layers.{i}.blocks.{j}.intermediate.dense.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.0.0.bias", f"backbone.encoder.layers.{i}.blocks.{j}.intermediate.dense.bias"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.1.weight", f"backbone.encoder.layers.{i}.blocks.{j}.output.dense.weight"))
            rename_keys.append((f"backbone.stages.{i}.blocks.{j}.ffn.layers.1.bias", f"backbone.encoder.layers.{i}.blocks.{j}.output.dense.bias"))

        if i < 3:
            rename_keys.append((f"backbone.stages.{i}.downsample.reduction.weight", f"backbone.encoder.layers.{i}.downsample.reduction.weight"))
            rename_keys.append((f"backbone.stages.{i}.downsample.norm.weight", f"backbone.encoder.layers.{i}.downsample.norm.weight"))
            rename_keys.append((f"backbone.stages.{i}.downsample.norm.bias", f"backbone.encoder.layers.{i}.downsample.norm.bias"))
        rename_keys.append((f"backbone.norm{i}.weight", f"backbone.hidden_states_norms.stage{i+1}.weight"))
        rename_keys.append((f"backbone.norm{i}.bias", f"backbone.hidden_states_norms.stage{i+1}.bias"))

    rename_keys.extend(
        [
            ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
            ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
            ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
            ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
        ]
    )

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def read_in_q_k_v(state_dict, backbone_config):
    num_features = [int(backbone_config.embed_dim * 2**i) for i in range(len(backbone_config.depths))]
    for i in range(len(backbone_config.depths)):
        dim = num_features[i]
        for j in range(backbone_config.depths[i]):
            in_proj_weight = state_dict.pop(f"backbone.stages.{i}.blocks.{j}.attn.w_msa.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.stages.{i}.blocks.{j}.attn.w_msa.qkv.bias")
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"] = in_proj_bias[: dim]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[
                dim : dim * 2, :
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.key.bias"] = in_proj_bias[
                dim : dim * 2
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[
                -dim :, :
            ]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"] = in_proj_bias[-dim :]


def correct_unfold_reduction_order(x):
    pass


def reverse_correct_unfold_reduction_order(x):
    pass


def correct_unfold_norm_order(x):
    pass


def reverse_correct_unfold_norm_order(x):
    pass


def convert_upernet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="upernet-swin-tiny",
        type=str,
        choices=[f"upernet-swin-{size}" for size in ["tiny", "small", "base", "large"]],
        help="Name of the Swin + UperNet model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_upernet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
