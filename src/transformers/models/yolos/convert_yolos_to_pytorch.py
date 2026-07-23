
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import YolosConfig, YolosForObjectDetection, YolosImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_yolos_config(yolos_name: str) -> YolosConfig:
    pass


def read_in_q_k_v(state_dict: dict, config: YolosConfig, base_model: bool = False):
    for i in range(config.num_hidden_layers):
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        state_dict[f"encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def rename_key(name: str) -> str:
    if "backbone" in name:
        name = name.replace("backbone", "vit")
    if "cls_token" in name:
        name = name.replace("cls_token", "embeddings.cls_token")
    if "det_token" in name:
        name = name.replace("det_token", "embeddings.detection_tokens")
    if "mid_pos_embed" in name:
        name = name.replace("mid_pos_embed", "encoder.mid_position_embeddings")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "embeddings.position_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "blocks" in name:
        name = name.replace("blocks", "encoder.layer")
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
    if "class_embed" in name:
        name = name.replace("class_embed", "class_labels_classifier")
    if "bbox_embed" in name:
        name = name.replace("bbox_embed", "bbox_predictor")
    if "vit.norm" in name:
        name = name.replace("vit.norm", "vit.layernorm")

    return name


def convert_state_dict(orig_state_dict: dict, model: YolosForObjectDetection) -> dict:
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[2])
            dim = model.vit.encoder.layer[layer_num].attention.attention.all_head_size
            if "weight" in key:
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.key.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.query.bias"] = val[:dim]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.key.bias"] = val[dim : dim * 2]
                orig_state_dict[f"vit.encoder.layer.{layer_num}.attention.attention.value.bias"] = val[-dim:]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def prepare_img() -> torch.Tensor:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_yolos_checkpoint(
    yolos_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolos_name",
        default="yolos_s_200_pre",
        type=str,
        help=(
            "Name of the YOLOS model you'd like to convert. Should be one of 'yolos_ti', 'yolos_s_200_pre',"
            " 'yolos_s_300_pre', 'yolos_s_dWr', 'yolos_base'."
        ),
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original state dict (.pth file)."
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
    convert_yolos_checkpoint(args.yolos_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
