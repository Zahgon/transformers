
import argparse
import json
from io import BytesIO

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import ConvNextConfig, SegformerImageProcessor, UperNetConfig, UperNetForSemanticSegmentation


def get_upernet_config(model_name):
    pass


def create_rename_keys(config):
    rename_keys = []

    rename_keys.append(("backbone.downsample_layers.0.0.weight", "backbone.embeddings.patch_embeddings.weight"))
    rename_keys.append(("backbone.downsample_layers.0.0.bias", "backbone.embeddings.patch_embeddings.bias"))
    rename_keys.append(("backbone.downsample_layers.0.1.weight", "backbone.embeddings.layernorm.weight"))
    rename_keys.append(("backbone.downsample_layers.0.1.bias", "backbone.embeddings.layernorm.bias"))
    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            rename_keys.append((f"backbone.stages.{i}.{j}.gamma", f"backbone.encoder.stages.{i}.layers.{j}.layer_scale_parameter"))
            rename_keys.append((f"backbone.stages.{i}.{j}.depthwise_conv.weight", f"backbone.encoder.stages.{i}.layers.{j}.dwconv.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.depthwise_conv.bias", f"backbone.encoder.stages.{i}.layers.{j}.dwconv.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.norm.weight", f"backbone.encoder.stages.{i}.layers.{j}.layernorm.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.norm.bias", f"backbone.encoder.stages.{i}.layers.{j}.layernorm.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv1.weight", f"backbone.encoder.stages.{i}.layers.{j}.pwconv1.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv1.bias", f"backbone.encoder.stages.{i}.layers.{j}.pwconv1.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv2.weight", f"backbone.encoder.stages.{i}.layers.{j}.pwconv2.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv2.bias", f"backbone.encoder.stages.{i}.layers.{j}.pwconv2.bias"))
        if i > 0:
            rename_keys.append((f"backbone.downsample_layers.{i}.0.weight", f"backbone.encoder.stages.{i}.downsampling_layer.0.weight"))
            rename_keys.append((f"backbone.downsample_layers.{i}.0.bias", f"backbone.encoder.stages.{i}.downsampling_layer.0.bias"))
            rename_keys.append((f"backbone.downsample_layers.{i}.1.weight", f"backbone.encoder.stages.{i}.downsampling_layer.1.weight"))
            rename_keys.append((f"backbone.downsample_layers.{i}.1.bias", f"backbone.encoder.stages.{i}.downsampling_layer.1.bias"))

        rename_keys.append((f"backbone.norm{i}.weight", f"backbone.hidden_states_norms.stage{i+1}.weight"))
        rename_keys.append((f"backbone.norm{i}.bias", f"backbone.hidden_states_norms.stage{i+1}.bias"))

    rename_keys.extend(
        [
            ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
            ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
            ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
            ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
        ]
    )

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def convert_upernet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="upernet-convnext-tiny",
        type=str,
        choices=[f"upernet-convnext-{size}" for size in ["tiny", "small", "base", "large", "xlarge"]],
        help="Name of the ConvNext UperNet model you'd like to convert.",
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
    convert_upernet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
