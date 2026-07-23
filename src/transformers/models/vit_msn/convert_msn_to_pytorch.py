
import argparse
import json
from io import BytesIO

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ViTImageProcessor, ViTMSNConfig, ViTMSNModel
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


torch.set_grad_enabled(False)


def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        rename_keys.append((f"module.blocks.{i}.norm1.weight", f"vit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"module.blocks.{i}.norm1.bias", f"vit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"module.blocks.{i}.attn.proj.weight", f"vit.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append((f"module.blocks.{i}.attn.proj.bias", f"vit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"module.blocks.{i}.norm2.weight", f"vit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"module.blocks.{i}.norm2.bias", f"vit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc1.weight", f"vit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc1.bias", f"vit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc2.weight", f"vit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"module.blocks.{i}.mlp.fc2.bias", f"vit.encoder.layer.{i}.output.dense.bias"))

    rename_keys.extend(
        [
            ("module.cls_token", "vit.embeddings.cls_token"),
            ("module.patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"),
            ("module.patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"),
            ("module.pos_embed", "vit.embeddings.position_embeddings"),
        ]
    )

    if base_model:
        rename_keys.extend(
            [
                ("module.norm.weight", "layernorm.weight"),
                ("module.norm.bias", "layernorm.bias"),
            ]
        )

        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    else:
        rename_keys.extend(
            [
                ("norm.weight", "vit.layernorm.weight"),
                ("norm.bias", "vit.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    return rename_keys


def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        in_proj_weight = state_dict.pop(f"module.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"module.blocks.{i}.attn.qkv.bias")
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def remove_classification_head_(state_dict):
    pass


def remove_projection_head(state_dict):
    pass


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def convert_vit_msn_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vit_msn_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
