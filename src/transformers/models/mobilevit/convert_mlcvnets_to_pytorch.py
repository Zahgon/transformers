
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    MobileViTConfig,
    MobileViTForImageClassification,
    MobileViTForSemanticSegmentation,
    MobileViTImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_mobilevit_config(mobilevit_name):
    pass


def rename_key(name, base_model=False):
    for i in range(1, 6):
        if f"layer_{i}." in name:
            name = name.replace(f"layer_{i}.", f"encoder.layer.{i - 1}.")

    if "conv_1." in name:
        name = name.replace("conv_1.", "conv_stem.")
    if ".block." in name:
        name = name.replace(".block.", ".")

    if "exp_1x1" in name:
        name = name.replace("exp_1x1", "expand_1x1")
    if "red_1x1" in name:
        name = name.replace("red_1x1", "reduce_1x1")
    if ".local_rep.conv_3x3." in name:
        name = name.replace(".local_rep.conv_3x3.", ".conv_kxk.")
    if ".local_rep.conv_1x1." in name:
        name = name.replace(".local_rep.conv_1x1.", ".conv_1x1.")
    if ".norm." in name:
        name = name.replace(".norm.", ".normalization.")
    if ".conv." in name:
        name = name.replace(".conv.", ".convolution.")
    if ".conv_proj." in name:
        name = name.replace(".conv_proj.", ".conv_projection.")

    for i in range(0, 2):
        for j in range(0, 4):
            if f".{i}.{j}." in name:
                name = name.replace(f".{i}.{j}.", f".{i}.layer.{j}.")

    for i in range(2, 6):
        for j in range(0, 4):
            if f".{i}.{j}." in name:
                name = name.replace(f".{i}.{j}.", f".{i}.")
                if "expand_1x1" in name:
                    name = name.replace("expand_1x1", "downsampling_layer.expand_1x1")
                if "conv_3x3" in name:
                    name = name.replace("conv_3x3", "downsampling_layer.conv_3x3")
                if "reduce_1x1" in name:
                    name = name.replace("reduce_1x1", "downsampling_layer.reduce_1x1")

    for i in range(2, 5):
        if f".global_rep.{i}.weight" in name:
            name = name.replace(f".global_rep.{i}.weight", ".layernorm.weight")
        if f".global_rep.{i}.bias" in name:
            name = name.replace(f".global_rep.{i}.bias", ".layernorm.bias")

    if ".global_rep." in name:
        name = name.replace(".global_rep.", ".transformer.")
    if ".pre_norm_mha.0." in name:
        name = name.replace(".pre_norm_mha.0.", ".layernorm_before.")
    if ".pre_norm_mha.1.out_proj." in name:
        name = name.replace(".pre_norm_mha.1.out_proj.", ".attention.output.dense.")
    if ".pre_norm_ffn.0." in name:
        name = name.replace(".pre_norm_ffn.0.", ".layernorm_after.")
    if ".pre_norm_ffn.1." in name:
        name = name.replace(".pre_norm_ffn.1.", ".intermediate.dense.")
    if ".pre_norm_ffn.4." in name:
        name = name.replace(".pre_norm_ffn.4.", ".output.dense.")
    if ".transformer." in name:
        name = name.replace(".transformer.", ".transformer.layer.")

    if ".aspp_layer." in name:
        name = name.replace(".aspp_layer.", ".")
    if ".aspp_pool." in name:
        name = name.replace(".aspp_pool.", ".")
    if "seg_head." in name:
        name = name.replace("seg_head.", "segmentation_head.")
    if "segmentation_head.classifier.classifier." in name:
        name = name.replace("segmentation_head.classifier.classifier.", "segmentation_head.classifier.")

    if "classifier.fc." in name:
        name = name.replace("classifier.fc.", "classifier.")
    elif (not base_model) and ("segmentation_head." not in name):
        name = "mobilevit." + name

    return name


def convert_state_dict(orig_state_dict, model, base_model=False):
    if base_model:
        model_prefix = ""
    else:
        model_prefix = "mobilevit."

    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        if key[:8] == "encoder.":
            key = key[8:]

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[0][6:]) - 1
            transformer_num = int(key_split[3])
            layer = model.get_submodule(f"{model_prefix}encoder.layer.{layer_num}")
            dim = layer.transformer.layer[transformer_num].attention.attention.all_head_size
            prefix = (
                f"{model_prefix}encoder.layer.{layer_num}.transformer.layer.{transformer_num}.attention.attention."
            )
            if "weight" in key:
                orig_state_dict[prefix + "query.weight"] = val[:dim, :]
                orig_state_dict[prefix + "key.weight"] = val[dim : dim * 2, :]
                orig_state_dict[prefix + "value.weight"] = val[-dim:, :]
            else:
                orig_state_dict[prefix + "query.bias"] = val[:dim]
                orig_state_dict[prefix + "key.bias"] = val[dim : dim * 2]
                orig_state_dict[prefix + "value.bias"] = val[-dim:]
        else:
            orig_state_dict[rename_key(key, base_model)] = val

    return orig_state_dict


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_movilevit_checkpoint(mobilevit_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mobilevit_name",
        default="mobilevit_s",
        type=str,
        help=(
            "Name of the MobileViT model you'd like to convert. Should be one of 'mobilevit_s', 'mobilevit_xs',"
            " 'mobilevit_xxs', 'deeplabv3_mobilevit_s', 'deeplabv3_mobilevit_xs', 'deeplabv3_mobilevit_xxs'."
        ),
    )
    parser.add_argument(
        "--checkpoint_path", required=True, type=str, help="Path to the original state dict (.pt file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_movilevit_checkpoint(
        args.mobilevit_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
