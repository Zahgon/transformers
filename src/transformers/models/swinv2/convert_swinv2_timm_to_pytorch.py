
import argparse
import json
from io import BytesIO

import httpx
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import AutoImageProcessor, Swinv2Config, Swinv2ForImageClassification


def get_swinv2_config(swinv2_name):
    pass


def rename_key(name):
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    if "layers" in name:
        name = "encoder." + name
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "q_bias" in name:
        name = name.replace("q_bias", "query.bias")
    if "k_bias" in name:
        name = name.replace("k_bias", "key.bias")
    if "v_bias" in name:
        name = name.replace("v_bias", "value.bias")
    if "cpb_mlp" in name:
        name = name.replace("cpb_mlp", "continuous_position_bias_mlp")
    if name == "norm.weight":
        name = "layernorm.weight"
    if name == "norm.bias":
        name = "layernorm.bias"

    if "head" in name:
        name = name.replace("head", "classifier")
    else:
        name = "swinv2." + name

    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        if "mask" in key:
            continue
        elif "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            block_num = int(key_split[3])
            dim = model.swinv2.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            if "weight" in key:
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"] = (
                    val[dim : dim * 2, :]
                )
                orig_state_dict[
                    f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"] = (
                    val[:dim]
                )
                orig_state_dict[f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"] = val[
                    dim : dim * 2
                ]
                orig_state_dict[f"swinv2.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"] = (
                    val[-dim:]
                )
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_swinv2_checkpoint(swinv2_name, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--swinv2_name",
        default="swinv2_tiny_patch4_window8_256",
        type=str,
        help="Name of the Swinv2 timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_swinv2_checkpoint(args.swinv2_name, args.pytorch_dump_folder_path)
