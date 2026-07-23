
import argparse
import json
import os
from io import BytesIO

import httpx
import numpy as np
import PIL
import tensorflow.keras.applications.efficientnet as efficientnet
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.preprocessing import image

from transformers import (
    EfficientNetConfig,
    EfficientNetForImageClassification,
    EfficientNetImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

model_classes = {
    "b0": efficientnet.EfficientNetB0,
    "b1": efficientnet.EfficientNetB1,
    "b2": efficientnet.EfficientNetB2,
    "b3": efficientnet.EfficientNetB3,
    "b4": efficientnet.EfficientNetB4,
    "b5": efficientnet.EfficientNetB5,
    "b6": efficientnet.EfficientNetB6,
    "b7": efficientnet.EfficientNetB7,
}

CONFIG_MAP = {
    "b0": {
        "hidden_dim": 1280,
        "width_coef": 1.0,
        "depth_coef": 1.0,
        "image_size": 224,
        "dropout_rate": 0.2,
        "dw_padding": [],
    },
    "b1": {
        "hidden_dim": 1280,
        "width_coef": 1.0,
        "depth_coef": 1.1,
        "image_size": 240,
        "dropout_rate": 0.2,
        "dw_padding": [16],
    },
    "b2": {
        "hidden_dim": 1408,
        "width_coef": 1.1,
        "depth_coef": 1.2,
        "image_size": 260,
        "dropout_rate": 0.3,
        "dw_padding": [5, 8, 16],
    },
    "b3": {
        "hidden_dim": 1536,
        "width_coef": 1.2,
        "depth_coef": 1.4,
        "image_size": 300,
        "dropout_rate": 0.3,
        "dw_padding": [5, 18],
    },
    "b4": {
        "hidden_dim": 1792,
        "width_coef": 1.4,
        "depth_coef": 1.8,
        "image_size": 380,
        "dropout_rate": 0.4,
        "dw_padding": [6],
    },
    "b5": {
        "hidden_dim": 2048,
        "width_coef": 1.6,
        "depth_coef": 2.2,
        "image_size": 456,
        "dropout_rate": 0.4,
        "dw_padding": [13, 27],
    },
    "b6": {
        "hidden_dim": 2304,
        "width_coef": 1.8,
        "depth_coef": 2.6,
        "image_size": 528,
        "dropout_rate": 0.5,
        "dw_padding": [31],
    },
    "b7": {
        "hidden_dim": 2560,
        "width_coef": 2.0,
        "depth_coef": 3.1,
        "image_size": 600,
        "dropout_rate": 0.5,
        "dw_padding": [18],
    },
}


def get_efficientnet_config(model_name):
    pass


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


def convert_image_processor(model_name):
    pass


def rename_keys(original_param_names):
    block_names = [v.split("_")[0].split("block")[1] for v in original_param_names if v.startswith("block")]
    block_names = sorted(set(block_names))
    num_blocks = len(block_names)
    block_name_mapping = {b: str(i) for b, i in zip(block_names, range(num_blocks))}

    rename_keys = []
    rename_keys.append(("stem_conv/kernel:0", "embeddings.convolution.weight"))
    rename_keys.append(("stem_bn/gamma:0", "embeddings.batchnorm.weight"))
    rename_keys.append(("stem_bn/beta:0", "embeddings.batchnorm.bias"))
    rename_keys.append(("stem_bn/moving_mean:0", "embeddings.batchnorm.running_mean"))
    rename_keys.append(("stem_bn/moving_variance:0", "embeddings.batchnorm.running_var"))

    for b in block_names:
        hf_b = block_name_mapping[b]
        rename_keys.append((f"block{b}_expand_conv/kernel:0", f"encoder.blocks.{hf_b}.expansion.expand_conv.weight"))
        rename_keys.append((f"block{b}_expand_bn/gamma:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.weight"))
        rename_keys.append((f"block{b}_expand_bn/beta:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.bias"))
        rename_keys.append(
            (f"block{b}_expand_bn/moving_mean:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_mean")
        )
        rename_keys.append(
            (f"block{b}_expand_bn/moving_variance:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_var")
        )
        rename_keys.append(
            (f"block{b}_dwconv/depthwise_kernel:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_conv.weight")
        )
        rename_keys.append((f"block{b}_bn/gamma:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.weight"))
        rename_keys.append((f"block{b}_bn/beta:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.bias"))
        rename_keys.append(
            (f"block{b}_bn/moving_mean:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_mean")
        )
        rename_keys.append(
            (f"block{b}_bn/moving_variance:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_var")
        )

        rename_keys.append((f"block{b}_se_reduce/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.weight"))
        rename_keys.append((f"block{b}_se_reduce/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.bias"))
        rename_keys.append((f"block{b}_se_expand/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.weight"))
        rename_keys.append((f"block{b}_se_expand/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.bias"))
        rename_keys.append(
            (f"block{b}_project_conv/kernel:0", f"encoder.blocks.{hf_b}.projection.project_conv.weight")
        )
        rename_keys.append((f"block{b}_project_bn/gamma:0", f"encoder.blocks.{hf_b}.projection.project_bn.weight"))
        rename_keys.append((f"block{b}_project_bn/beta:0", f"encoder.blocks.{hf_b}.projection.project_bn.bias"))
        rename_keys.append(
            (f"block{b}_project_bn/moving_mean:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_mean")
        )
        rename_keys.append(
            (f"block{b}_project_bn/moving_variance:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_var")
        )

    rename_keys.append(("top_conv/kernel:0", "encoder.top_conv.weight"))
    rename_keys.append(("top_bn/gamma:0", "encoder.top_bn.weight"))
    rename_keys.append(("top_bn/beta:0", "encoder.top_bn.bias"))
    rename_keys.append(("top_bn/moving_mean:0", "encoder.top_bn.running_mean"))
    rename_keys.append(("top_bn/moving_variance:0", "encoder.top_bn.running_var"))

    key_mapping = {}
    for item in rename_keys:
        if item[0] in original_param_names:
            key_mapping[item[0]] = "efficientnet." + item[1]

    key_mapping["predictions/kernel:0"] = "classifier.weight"
    key_mapping["predictions/bias:0"] = "classifier.bias"
    return key_mapping


def replace_params(hf_params, tf_params, key_mapping):
    pass


@torch.no_grad()
def convert_efficientnet_checkpoint(model_name, pytorch_dump_folder_path, save_model, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="b0",
        type=str,
        help="Version name of the EfficientNet model you want to convert, select from [b0, b1, b2, b3, b4, b5, b6, b7].",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="hf_model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")

    args = parser.parse_args()
    convert_efficientnet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub)
