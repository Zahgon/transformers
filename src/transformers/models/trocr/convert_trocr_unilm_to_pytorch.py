
import argparse
from io import BytesIO
from pathlib import Path

import httpx
import torch
from PIL import Image

from transformers import (
    RobertaTokenizer,
    TrOCRConfig,
    TrOCRForCausalLM,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTConfig,
    ViTImageProcessor,
    ViTModel,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def create_rename_keys(encoder_config, decoder_config):
    rename_keys = []
    for i in range(encoder_config.num_hidden_layers):
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm1.weight", f"encoder.encoder.layer.{i}.layernorm_before.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm1.bias", f"encoder.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.weight", f"encoder.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.bias", f"encoder.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm2.weight", f"encoder.encoder.layer.{i}.layernorm_after.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm2.bias", f"encoder.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.weight", f"encoder.encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.bias", f"encoder.encoder.layer.{i}.intermediate.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc2.weight", f"encoder.encoder.layer.{i}.output.dense.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.mlp.fc2.bias", f"encoder.encoder.layer.{i}.output.dense.bias"))

    rename_keys.extend(
        [
            ("encoder.deit.cls_token", "encoder.embeddings.cls_token"),
            ("encoder.deit.pos_embed", "encoder.embeddings.position_embeddings"),
            ("encoder.deit.patch_embed.proj.weight", "encoder.embeddings.patch_embeddings.projection.weight"),
            ("encoder.deit.patch_embed.proj.bias", "encoder.embeddings.patch_embeddings.projection.bias"),
            ("encoder.deit.norm.weight", "encoder.layernorm.weight"),
            ("encoder.deit.norm.bias", "encoder.layernorm.bias"),
        ]
    )

    return rename_keys


def read_in_q_k_v(state_dict, encoder_config):
    for i in range(encoder_config.num_hidden_layers):
        in_proj_weight = state_dict.pop(f"encoder.deit.blocks.{i}.attn.qkv.weight")

        state_dict[f"encoder.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : encoder_config.hidden_size, :
        ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            encoder_config.hidden_size : encoder_config.hidden_size * 2, :
        ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -encoder_config.hidden_size :, :
        ]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def prepare_img(checkpoint_url):
    if "handwritten" in checkpoint_url:
        url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"  # industry
    elif "printed" in checkpoint_url or "stage1" in checkpoint_url:
        url = "https://www.researchgate.net/profile/Dinh-Sang/publication/338099565/figure/fig8/AS:840413229350922@1577381536857/An-receipt-example-in-the-SROIE-2019-dataset_Q640.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read())).convert("RGB")
    return image


@torch.no_grad()
def convert_tr_ocr_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_tr_ocr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
