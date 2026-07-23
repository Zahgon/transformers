
import argparse
from io import BytesIO

import httpx
import torch
from PIL import Image

from transformers import SwinConfig, SwinForMaskedImageModeling, ViTImageProcessor


def get_swin_config(model_name):
    pass


def rename_key(name):
    if "encoder.mask_token" in name:
        name = name.replace("encoder.mask_token", "embeddings.mask_token")
    if "encoder.patch_embed.proj" in name:
        name = name.replace("encoder.patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "encoder.patch_embed.norm" in name:
        name = name.replace("encoder.patch_embed.norm", "embeddings.norm")
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

    if name == "encoder.norm.weight":
        name = "layernorm.weight"
    if name == "encoder.norm.bias":
        name = "layernorm.bias"

    if "decoder" in name:
        pass
    else:
        name = "swin." + name

    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        if "attn_mask" in key:
            pass
        elif "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[2])
            block_num = int(key_split[4])
            dim = model.swin.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            if "weight" in key:
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"] = (
                    val[:dim, :]
                )
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"] = (
                    val[-dim:, :]
                )
            else:
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"] = val[
                    :dim
                ]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"] = val[
                    dim : dim * 2
                ]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"] = val[
                    -dim:
                ]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_swin_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="swin-base-simmim-window6-192",
        type=str,
        choices=["swin-base-simmim-window6-192", "swin-large-simmim-window12-192"],
        help="Name of the Swin SimMIM model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/SwinSimMIM/simmim_pretrain__swin_base__img192_window6__100ep.pth",
        type=str,
        help="Path to the original PyTorch checkpoint (.pth file).",
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
    convert_swin_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
