
import argparse
from io import BytesIO
from pathlib import Path

import httpx
import torch
from PIL import Image

from transformers import PvtImageProcessor, PvtV2Config, PvtV2ForImageClassification
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def create_rename_keys(config):
    rename_keys = []
    for i in range(config.num_encoder_blocks):
        rename_keys.append(
            (f"patch_embed{i + 1}.proj.weight", f"pvt_v2.encoder.layers.{i}.patch_embedding.proj.weight")
        )
        rename_keys.append((f"patch_embed{i + 1}.proj.bias", f"pvt_v2.encoder.layers.{i}.patch_embedding.proj.bias"))
        rename_keys.append(
            (f"patch_embed{i + 1}.norm.weight", f"pvt_v2.encoder.layers.{i}.patch_embedding.layer_norm.weight")
        )
        rename_keys.append(
            (f"patch_embed{i + 1}.norm.bias", f"pvt_v2.encoder.layers.{i}.patch_embedding.layer_norm.bias")
        )
        rename_keys.append((f"norm{i + 1}.weight", f"pvt_v2.encoder.layers.{i}.layer_norm.weight"))
        rename_keys.append((f"norm{i + 1}.bias", f"pvt_v2.encoder.layers.{i}.layer_norm.bias"))

        for j in range(config.depths[i]):
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.q.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.query.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.q.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.query.bias")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.kv.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.kv.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.kv.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.kv.bias")
            )

            if config.linear_attention or config.sr_ratios[i] > 1:
                rename_keys.append(
                    (
                        f"block{i + 1}.{j}.attn.norm.weight",
                        f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.layer_norm.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"block{i + 1}.{j}.attn.norm.bias",
                        f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.layer_norm.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"block{i + 1}.{j}.attn.sr.weight",
                        f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.spatial_reduction.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"block{i + 1}.{j}.attn.sr.bias",
                        f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.spatial_reduction.bias",
                    )
                )

            rename_keys.append(
                (f"block{i + 1}.{j}.attn.proj.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.proj.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.proj.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.attention.proj.bias")
            )

            rename_keys.append(
                (f"block{i + 1}.{j}.norm1.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.layer_norm_1.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.norm1.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.layer_norm_1.bias")
            )

            rename_keys.append(
                (f"block{i + 1}.{j}.norm2.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.layer_norm_2.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.norm2.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.layer_norm_2.bias")
            )

            rename_keys.append(
                (f"block{i + 1}.{j}.mlp.fc1.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dense1.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.mlp.fc1.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dense1.bias")
            )
            rename_keys.append(
                (
                    f"block{i + 1}.{j}.mlp.dwconv.dwconv.weight",
                    f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dwconv.dwconv.weight",
                )
            )
            rename_keys.append(
                (
                    f"block{i + 1}.{j}.mlp.dwconv.dwconv.bias",
                    f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dwconv.dwconv.bias",
                )
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.mlp.fc2.weight", f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dense2.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.mlp.fc2.bias", f"pvt_v2.encoder.layers.{i}.blocks.{j}.mlp.dense2.bias")
            )

    rename_keys.extend(
        [
            ("head.weight", "classifier.weight"),
            ("head.bias", "classifier.bias"),
        ]
    )

    return rename_keys


def read_in_k_v(state_dict, config):
    pass


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_pvt_v2_checkpoint(pvt_v2_size, pvt_v2_checkpoint, pytorch_dump_folder_path, verify_imagenet_weights=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pvt_v2_size",
        default="b0",
        type=str,
        help="Size of the PVTv2 pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pvt_v2_checkpoint",
        default="pvt_v2_b0.pth",
        type=str,
        help="Checkpoint of the PVTv2 pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--verify-imagenet-weights",
        action="store_true",
        default=False,
        help="Verifies the correct conversion of author-published pretrained ImageNet weights.",
    )

    args = parser.parse_args()
    convert_pvt_v2_checkpoint(
        pvt_v2_size=args.pvt_v2_size,
        pvt_v2_checkpoint=args.pvt_v2_checkpoint,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        verify_imagenet_weights=args.verify_imagenet_weights,
    )
