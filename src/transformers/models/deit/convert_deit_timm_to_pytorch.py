
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import DeiTConfig, DeiTForImageClassificationWithTeacher, DeiTImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        rename_keys.append((f"blocks.{i}.norm1.weight", f"deit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"deit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"deit.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"deit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"deit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"deit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"deit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"deit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"deit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"deit.encoder.layer.{i}.output.dense.bias"))

    rename_keys.extend(
        [
            ("cls_token", "deit.embeddings.cls_token"),
            ("dist_token", "deit.embeddings.distillation_token"),
            ("patch_embed.proj.weight", "deit.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "deit.embeddings.patch_embeddings.projection.bias"),
            ("pos_embed", "deit.embeddings.position_embeddings"),
        ]
    )

    if base_model:
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
                ("pre_logits.fc.weight", "pooler.dense.weight"),
                ("pre_logits.fc.bias", "pooler.dense.bias"),
            ]
        )

        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("deit") else pair for pair in rename_keys]
    else:
        rename_keys.extend(
            [
                ("norm.weight", "deit.layernorm.weight"),
                ("norm.bias", "deit.layernorm.bias"),
                ("head.weight", "cls_classifier.weight"),
                ("head.bias", "cls_classifier.bias"),
                ("head_dist.weight", "distillation_classifier.weight"),
                ("head_dist.bias", "distillation_classifier.bias"),
            ]
        )

    return rename_keys


def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "deit."
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
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


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_deit_checkpoint(deit_name, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deit_name",
        default="vit_deit_base_distilled_patch16_224",
        type=str,
        help="Name of the DeiT timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_deit_checkpoint(args.deit_name, args.pytorch_dump_folder_path)
