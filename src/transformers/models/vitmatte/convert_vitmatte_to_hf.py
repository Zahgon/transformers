
import argparse
from io import BytesIO

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import VitDetConfig, VitMatteConfig, VitMatteForImageMatting, VitMatteImageProcessor


def get_config(model_name):
    hidden_size = 384 if "small" in model_name else 768
    num_attention_heads = 6 if "small" in model_name else 12

    backbone_config = VitDetConfig(
        num_channels=4,
        image_size=512,
        pretrain_image_size=224,
        patch_size=16,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_absolute_position_embeddings=True,
        use_relative_position_embeddings=True,
        window_size=14,
        window_block_indices=[0, 1, 3, 4, 6, 7, 9, 10],
        residual_block_indices=[2, 5, 8, 11],
        out_features=["stage12"],
    )

    return VitMatteConfig(backbone_config=backbone_config, hidden_size=hidden_size)


def create_rename_keys(config):
    rename_keys = []

    rename_keys.append(("backbone.pos_embed", "backbone.embeddings.position_embeddings"))
    rename_keys.append(("backbone.patch_embed.proj.weight", "backbone.embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.proj.bias", "backbone.embeddings.projection.bias"))

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def convert_vitmatte_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="vitmatte-small-composition-1k",
        type=str,
        choices=[
            "vitmatte-small-composition-1k",
            "vitmatte-base-composition-1k",
            "vitmatte-small-distinctions-646",
            "vitmatte-base-distinctions-646",
        ],
        help="Name of the VitMatte model you'd like to convert.",
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
    convert_vitmatte_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
