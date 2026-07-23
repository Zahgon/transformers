
import argparse
import collections
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    MobileViTImageProcessor,
    MobileViTV2Config,
    MobileViTV2ForImageClassification,
    MobileViTV2ForSemanticSegmentation,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_orig_config_file(orig_cfg_file):
    pass


def get_mobilevitv2_config(task_name, orig_cfg_file):
    pass


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def create_rename_keys(state_dict, base_model=False):
    if base_model:
        model_prefix = ""
    else:
        model_prefix = "mobilevitv2."

    rename_keys = []
    for k in state_dict:
        if k[:8] == "encoder.":
            k_new = k[8:]
        else:
            k_new = k

        if ".block." in k:
            k_new = k_new.replace(".block.", ".")
        if ".conv." in k:
            k_new = k_new.replace(".conv.", ".convolution.")
        if ".norm." in k:
            k_new = k_new.replace(".norm.", ".normalization.")

        if "conv_1." in k:
            k_new = k_new.replace("conv_1.", f"{model_prefix}conv_stem.")
        for i in [1, 2]:
            if f"layer_{i}." in k:
                k_new = k_new.replace(f"layer_{i}.", f"{model_prefix}encoder.layer.{i - 1}.layer.")
        if ".exp_1x1." in k:
            k_new = k_new.replace(".exp_1x1.", ".expand_1x1.")
        if ".red_1x1." in k:
            k_new = k_new.replace(".red_1x1.", ".reduce_1x1.")

        for i in [3, 4, 5]:
            if f"layer_{i}.0." in k:
                k_new = k_new.replace(f"layer_{i}.0.", f"{model_prefix}encoder.layer.{i - 1}.downsampling_layer.")
            if f"layer_{i}.1.local_rep.0." in k:
                k_new = k_new.replace(f"layer_{i}.1.local_rep.0.", f"{model_prefix}encoder.layer.{i - 1}.conv_kxk.")
            if f"layer_{i}.1.local_rep.1." in k:
                k_new = k_new.replace(f"layer_{i}.1.local_rep.1.", f"{model_prefix}encoder.layer.{i - 1}.conv_1x1.")

        for i in [3, 4, 5]:
            if i == 3:
                j_in = [0, 1]
            elif i == 4:
                j_in = [0, 1, 2, 3]
            elif i == 5:
                j_in = [0, 1, 2]

            for j in j_in:
                if f"layer_{i}.1.global_rep.{j}." in k:
                    k_new = k_new.replace(
                        f"layer_{i}.1.global_rep.{j}.", f"{model_prefix}encoder.layer.{i - 1}.transformer.layer.{j}."
                    )
            if f"layer_{i}.1.global_rep.{j + 1}." in k:
                k_new = k_new.replace(
                    f"layer_{i}.1.global_rep.{j + 1}.", f"{model_prefix}encoder.layer.{i - 1}.layernorm."
                )

            if f"layer_{i}.1.conv_proj." in k:
                k_new = k_new.replace(
                    f"layer_{i}.1.conv_proj.", f"{model_prefix}encoder.layer.{i - 1}.conv_projection."
                )

        if "pre_norm_attn.0." in k:
            k_new = k_new.replace("pre_norm_attn.0.", "layernorm_before.")
        if "pre_norm_attn.1." in k:
            k_new = k_new.replace("pre_norm_attn.1.", "attention.")
        if "pre_norm_ffn.0." in k:
            k_new = k_new.replace("pre_norm_ffn.0.", "layernorm_after.")
        if "pre_norm_ffn.1." in k:
            k_new = k_new.replace("pre_norm_ffn.1.", "ffn.conv1.")
        if "pre_norm_ffn.3." in k:
            k_new = k_new.replace("pre_norm_ffn.3.", "ffn.conv2.")

        if "classifier.1." in k:
            k_new = k_new.replace("classifier.1.", "classifier.")

        if "seg_head." in k:
            k_new = k_new.replace("seg_head.", "segmentation_head.")
        if ".aspp_layer." in k:
            k_new = k_new.replace(".aspp_layer.", ".")
        if ".aspp_pool." in k:
            k_new = k_new.replace(".aspp_pool.", ".")

        rename_keys.append((k, k_new))
    return rename_keys


def remove_unused_keys(state_dict):
    pass


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_mobilevitv2_checkpoint(task_name, checkpoint_path, orig_config_path, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="imagenet1k_256",
        type=str,
        help=(
            "Name of the task for which the MobileViTV2 model you'd like to convert is trained on . "
            """
                Classification (ImageNet-1k)
                    - MobileViTV2 (256x256) : imagenet1k_256
                    - MobileViTV2 (Trained on 256x256 and Finetuned on 384x384) : imagenet1k_384
                    - MobileViTV2 (Trained on ImageNet-21k and Finetuned on ImageNet-1k 256x256) :
                      imagenet21k_to_1k_256
                    - MobileViTV2 (Trained on ImageNet-21k, Finetuned on ImageNet-1k 256x256, and Finetuned on
                      ImageNet-1k 384x384) : imagenet21k_to_1k_384
                Segmentation
                    - ADE20K Dataset : ade20k_deeplabv3
                    - Pascal VOC 2012 Dataset: voc_deeplabv3
            """
        ),
        choices=[
            "imagenet1k_256",
            "imagenet1k_384",
            "imagenet21k_to_1k_256",
            "imagenet21k_to_1k_384",
            "ade20k_deeplabv3",
            "voc_deeplabv3",
        ],
    )

    parser.add_argument(
        "--orig_checkpoint_path", required=True, type=str, help="Path to the original state dict (.pt file)."
    )
    parser.add_argument(
        "--orig_config_path",
        required=True,
        type=str,
        help="Path to the original config file. yaml.load will be used to load the file, please be wary of which file you're loading.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_mobilevitv2_checkpoint(
        args.task, args.orig_checkpoint_path, args.orig_config_path, args.pytorch_dump_folder_path
    )
