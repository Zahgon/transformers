
import argparse
from io import BytesIO
from pathlib import Path

import httpx
import torch
from PIL import Image

from transformers import PvtConfig, PvtForImageClassification, PvtImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def create_rename_keys(config):
    rename_keys = []
    for i in range(config.num_encoder_blocks):
        rename_keys.append((f"pos_embed{i + 1}", f"pvt.encoder.patch_embeddings.{i}.position_embeddings"))

        rename_keys.append((f"patch_embed{i + 1}.proj.weight", f"pvt.encoder.patch_embeddings.{i}.projection.weight"))
        rename_keys.append((f"patch_embed{i + 1}.proj.bias", f"pvt.encoder.patch_embeddings.{i}.projection.bias"))
        rename_keys.append((f"patch_embed{i + 1}.norm.weight", f"pvt.encoder.patch_embeddings.{i}.layer_norm.weight"))
        rename_keys.append((f"patch_embed{i + 1}.norm.bias", f"pvt.encoder.patch_embeddings.{i}.layer_norm.bias"))

        for j in range(config.depths[i]):
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.q.weight", f"pvt.encoder.block.{i}.{j}.attention.self.query.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.q.bias", f"pvt.encoder.block.{i}.{j}.attention.self.query.bias")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.kv.weight", f"pvt.encoder.block.{i}.{j}.attention.self.kv.weight")
            )
            rename_keys.append((f"block{i + 1}.{j}.attn.kv.bias", f"pvt.encoder.block.{i}.{j}.attention.self.kv.bias"))

            if config.sequence_reduction_ratios[i] > 1:
                rename_keys.append(
                    (
                        f"block{i + 1}.{j}.attn.norm.weight",
                        f"pvt.encoder.block.{i}.{j}.attention.self.layer_norm.weight",
                    )
                )
                rename_keys.append(
                    (f"block{i + 1}.{j}.attn.norm.bias", f"pvt.encoder.block.{i}.{j}.attention.self.layer_norm.bias")
                )
                rename_keys.append(
                    (
                        f"block{i + 1}.{j}.attn.sr.weight",
                        f"pvt.encoder.block.{i}.{j}.attention.self.sequence_reduction.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"block{i + 1}.{j}.attn.sr.bias",
                        f"pvt.encoder.block.{i}.{j}.attention.self.sequence_reduction.bias",
                    )
                )

            rename_keys.append(
                (f"block{i + 1}.{j}.attn.proj.weight", f"pvt.encoder.block.{i}.{j}.attention.output.dense.weight")
            )
            rename_keys.append(
                (f"block{i + 1}.{j}.attn.proj.bias", f"pvt.encoder.block.{i}.{j}.attention.output.dense.bias")
            )

            rename_keys.append((f"block{i + 1}.{j}.norm1.weight", f"pvt.encoder.block.{i}.{j}.layer_norm_1.weight"))
            rename_keys.append((f"block{i + 1}.{j}.norm1.bias", f"pvt.encoder.block.{i}.{j}.layer_norm_1.bias"))

            rename_keys.append((f"block{i + 1}.{j}.norm2.weight", f"pvt.encoder.block.{i}.{j}.layer_norm_2.weight"))
            rename_keys.append((f"block{i + 1}.{j}.norm2.bias", f"pvt.encoder.block.{i}.{j}.layer_norm_2.bias"))

            rename_keys.append((f"block{i + 1}.{j}.mlp.fc1.weight", f"pvt.encoder.block.{i}.{j}.mlp.dense1.weight"))
            rename_keys.append((f"block{i + 1}.{j}.mlp.fc1.bias", f"pvt.encoder.block.{i}.{j}.mlp.dense1.bias"))
            rename_keys.append((f"block{i + 1}.{j}.mlp.fc2.weight", f"pvt.encoder.block.{i}.{j}.mlp.dense2.weight"))
            rename_keys.append((f"block{i + 1}.{j}.mlp.fc2.bias", f"pvt.encoder.block.{i}.{j}.mlp.dense2.bias"))

    rename_keys.extend(
        [
            ("cls_token", "pvt.encoder.patch_embeddings.3.cls_token"),
        ]
    )
    rename_keys.extend(
        [
            ("norm.weight", "pvt.encoder.layer_norm.weight"),
            ("norm.bias", "pvt.encoder.layer_norm.bias"),
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
def convert_pvt_checkpoint(pvt_size, pvt_checkpoint, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pvt_size",
        default="tiny",
        type=str,
        help="Size of the PVT pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pvt_checkpoint",
        default="pvt_tiny.pth",
        type=str,
        help="Checkpoint of the PVT pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_pvt_checkpoint(args.pvt_size, args.pvt_checkpoint, args.pytorch_dump_folder_path)
