
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import BeitConfig, BeitForImageClassification, BeitForMaskedImageModeling, BeitImageProcessor
from transformers.image_utils import PILImageResampling
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def create_rename_keys(config, has_lm_head=False, is_semantic=False):
    prefix = "backbone." if is_semantic else ""

    rename_keys = []
    for i in range(config.num_hidden_layers):
        rename_keys.append((f"{prefix}blocks.{i}.norm1.weight", f"beit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm1.bias", f"beit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.weight", f"beit.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.bias", f"beit.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append((f"{prefix}blocks.{i}.norm2.weight", f"beit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm2.bias", f"beit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc1.weight", f"beit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc1.bias", f"beit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.weight", f"beit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.bias", f"beit.encoder.layer.{i}.output.dense.bias"))

    rename_keys.extend(
        [
            (f"{prefix}cls_token", "beit.embeddings.cls_token"),
            (f"{prefix}patch_embed.proj.weight", "beit.embeddings.patch_embeddings.projection.weight"),
            (f"{prefix}patch_embed.proj.bias", "beit.embeddings.patch_embeddings.projection.bias"),
            (f"{prefix}pos_embed", "beit.embeddings.position_embeddings"),
        ]
    )

    if has_lm_head:
        rename_keys.extend(
            [
                ("mask_token", "beit.embeddings.mask_token"),
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
            ]
        )
    else:
        rename_keys.extend(
            [
                ("fc_norm.weight", "beit.pooler.layernorm.weight"),
                ("fc_norm.bias", "beit.pooler.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    return rename_keys


def read_in_q_k_v(state_dict, config, has_lm_head=False, is_semantic=False):
    for i in range(config.num_hidden_layers):
        prefix = "backbone." if is_semantic else ""
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.v_bias")

        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"beit.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.bias"] = v_bias

        gamma_1 = state_dict.pop(f"{prefix}blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"{prefix}blocks.{i}.gamma_2")

        state_dict[f"beit.encoder.layer.{i}.lambda_1"] = gamma_1
        state_dict[f"beit.encoder.layer.{i}.lambda_2"] = gamma_2


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_dit_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    args = parser.parse_args()
    convert_dit_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
