
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    BertTokenizer,
    ViltConfig,
    ViltForImageAndTextRetrieval,
    ViltForImagesAndTextClassification,
    ViltForMaskedLM,
    ViltForQuestionAnswering,
    ViltImageProcessor,
    ViltProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def create_rename_keys(config, vqa_model=False, nlvr_model=False, irtr_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        rename_keys.append((f"transformer.blocks.{i}.norm1.weight", f"vilt.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"transformer.blocks.{i}.norm1.bias", f"vilt.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"transformer.blocks.{i}.attn.proj.weight", f"vilt.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"transformer.blocks.{i}.attn.proj.bias", f"vilt.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append((f"transformer.blocks.{i}.norm2.weight", f"vilt.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"transformer.blocks.{i}.norm2.bias", f"vilt.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"transformer.blocks.{i}.mlp.fc1.weight", f"vilt.encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc1.bias", f"vilt.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc2.weight", f"vilt.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc2.bias", f"vilt.encoder.layer.{i}.output.dense.bias"))

    rename_keys.extend(
        [
            ("text_embeddings.word_embeddings.weight", "vilt.embeddings.text_embeddings.word_embeddings.weight"),
            (
                "text_embeddings.position_embeddings.weight",
                "vilt.embeddings.text_embeddings.position_embeddings.weight",
            ),
            ("text_embeddings.position_ids", "vilt.embeddings.text_embeddings.position_ids"),
            (
                "text_embeddings.token_type_embeddings.weight",
                "vilt.embeddings.text_embeddings.token_type_embeddings.weight",
            ),
            ("text_embeddings.LayerNorm.weight", "vilt.embeddings.text_embeddings.LayerNorm.weight"),
            ("text_embeddings.LayerNorm.bias", "vilt.embeddings.text_embeddings.LayerNorm.bias"),
            ("transformer.cls_token", "vilt.embeddings.cls_token"),
            ("transformer.patch_embed.proj.weight", "vilt.embeddings.patch_embeddings.projection.weight"),
            ("transformer.patch_embed.proj.bias", "vilt.embeddings.patch_embeddings.projection.bias"),
            ("transformer.pos_embed", "vilt.embeddings.position_embeddings"),
            ("token_type_embeddings.weight", "vilt.embeddings.token_type_embeddings.weight"),
        ]
    )

    rename_keys.extend(
        [
            ("transformer.norm.weight", "vilt.layernorm.weight"),
            ("transformer.norm.bias", "vilt.layernorm.bias"),
            ("pooler.dense.weight", "vilt.pooler.dense.weight"),
            ("pooler.dense.bias", "vilt.pooler.dense.bias"),
        ]
    )

    if vqa_model:
        rename_keys.extend(
            [
                ("vqa_classifier.0.weight", "classifier.0.weight"),
                ("vqa_classifier.0.bias", "classifier.0.bias"),
                ("vqa_classifier.1.weight", "classifier.1.weight"),
                ("vqa_classifier.1.bias", "classifier.1.bias"),
                ("vqa_classifier.3.weight", "classifier.3.weight"),
                ("vqa_classifier.3.bias", "classifier.3.bias"),
            ]
        )
    elif nlvr_model:
        rename_keys.extend(
            [
                ("nlvr2_classifier.0.weight", "classifier.0.weight"),
                ("nlvr2_classifier.0.bias", "classifier.0.bias"),
                ("nlvr2_classifier.1.weight", "classifier.1.weight"),
                ("nlvr2_classifier.1.bias", "classifier.1.bias"),
                ("nlvr2_classifier.3.weight", "classifier.3.weight"),
                ("nlvr2_classifier.3.bias", "classifier.3.bias"),
            ]
        )
    else:
        pass

    return rename_keys


def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        prefix = "vilt."
        in_proj_weight = state_dict.pop(f"transformer.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"transformer.blocks.{i}.attn.qkv.bias")
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


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


@torch.no_grad()
def convert_vilt_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vilt_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
