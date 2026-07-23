
import argparse
import json
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    ConditionalDetrConfig,
    ConditionalDetrForObjectDetection,
    ConditionalDetrForSegmentation,
    ConditionalDetrImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

rename_keys = []
for i in range(6):
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.weight", f"encoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.bias", f"encoder.layers.{i}.self_attn.out_proj.bias")
    )
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"encoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"encoder.layers.{i}.fc1.bias"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"encoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"encoder.layers.{i}.fc2.bias"))
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.norm1.weight", f"encoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.encoder.layers.{i}.norm1.bias", f"encoder.layers.{i}.self_attn_layer_norm.bias"))
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.weight", f"encoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"encoder.layers.{i}.final_layer_norm.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"decoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.out_proj.bias")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.weight",
            f"decoder.layers.{i}.encoder_attn.out_proj.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.bias",
            f"decoder.layers.{i}.encoder_attn.out_proj.bias",
        )
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"decoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"decoder.layers.{i}.fc1.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"decoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"decoder.layers.{i}.fc2.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm1.weight", f"decoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm1.bias", f"decoder.layers.{i}.self_attn_layer_norm.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.encoder_attn_layer_norm.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.encoder_attn_layer_norm.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))

    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.sa_qcontent_proj.weight",
            f"decoder.layers.{i}.self_attn.q_content_proj.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.sa_kcontent_proj.weight",
            f"decoder.layers.{i}.self_attn.k_content_proj.weight",
        )
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qpos_proj.weight", f"decoder.layers.{i}.self_attn.q_pos_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kpos_proj.weight", f"decoder.layers.{i}.self_attn.k_pos_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_v_proj.weight", f"decoder.layers.{i}.self_attn.v_proj.weight")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_qcontent_proj.weight",
            f"decoder.layers.{i}.encoder_attn.q_content_proj.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_kcontent_proj.weight",
            f"decoder.layers.{i}.encoder_attn.k_content_proj.weight",
        )
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kpos_proj.weight", f"decoder.layers.{i}.encoder_attn.k_pos_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_v_proj.weight", f"decoder.layers.{i}.encoder_attn.v_proj.weight")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.weight",
            f"decoder.layers.{i}.encoder_attn.q_pos_sine_proj.weight",
        )
    )

    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qcontent_proj.bias", f"decoder.layers.{i}.self_attn.q_content_proj.bias")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kcontent_proj.bias", f"decoder.layers.{i}.self_attn.k_content_proj.bias")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qpos_proj.bias", f"decoder.layers.{i}.self_attn.q_pos_proj.bias")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kpos_proj.bias", f"decoder.layers.{i}.self_attn.k_pos_proj.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_v_proj.bias", f"decoder.layers.{i}.self_attn.v_proj.bias"))
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_qcontent_proj.bias",
            f"decoder.layers.{i}.encoder_attn.q_content_proj.bias",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_kcontent_proj.bias",
            f"decoder.layers.{i}.encoder_attn.k_content_proj.bias",
        )
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kpos_proj.bias", f"decoder.layers.{i}.encoder_attn.k_pos_proj.bias")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_v_proj.bias", f"decoder.layers.{i}.encoder_attn.v_proj.bias")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.bias",
            f"decoder.layers.{i}.encoder_attn.q_pos_sine_proj.bias",
        )
    )

rename_keys.extend(
    [
        ("input_proj.weight", "input_projection.weight"),
        ("input_proj.bias", "input_projection.bias"),
        ("query_embed.weight", "query_position_embeddings.weight"),
        ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),
        ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),
        ("class_embed.weight", "class_labels_classifier.weight"),
        ("class_embed.bias", "class_labels_classifier.bias"),
        ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),
        ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),
        ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),
        ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),
        ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),
        ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),
        ("transformer.decoder.ref_point_head.layers.0.weight", "decoder.ref_point_head.layers.0.weight"),
        ("transformer.decoder.ref_point_head.layers.0.bias", "decoder.ref_point_head.layers.0.bias"),
        ("transformer.decoder.ref_point_head.layers.1.weight", "decoder.ref_point_head.layers.1.weight"),
        ("transformer.decoder.ref_point_head.layers.1.bias", "decoder.ref_point_head.layers.1.bias"),
        ("transformer.decoder.query_scale.layers.0.weight", "decoder.query_scale.layers.0.weight"),
        ("transformer.decoder.query_scale.layers.0.bias", "decoder.query_scale.layers.0.bias"),
        ("transformer.decoder.query_scale.layers.1.weight", "decoder.query_scale.layers.1.weight"),
        ("transformer.decoder.query_scale.layers.1.bias", "decoder.query_scale.layers.1.bias"),
        ("transformer.decoder.layers.0.ca_qpos_proj.weight", "decoder.layers.0.encoder_attn.q_pos_proj.weight"),
        ("transformer.decoder.layers.0.ca_qpos_proj.bias", "decoder.layers.0.encoder_attn.q_pos_proj.bias"),
    ]
)


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def rename_backbone_keys(state_dict):
    pass


def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""
    if is_panoptic:
        prefix = "conditional_detr."

    for i in range(6):
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))

    return image


@torch.no_grad()
def convert_conditional_detr_checkpoint(model_name, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="conditional_detr_resnet50",
        type=str,
        help="Name of the CONDITIONAL_DETR model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_conditional_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path)
