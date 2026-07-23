
import argparse
import json
import os
import pickle
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, MaskFormerImageProcessor, ResNetConfig
from transformers.utils import logging

from ...utils import strtobool


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_maskformer_config(model_name: str):
    pass


def create_rename_keys(config):
    rename_keys = []
    rename_keys.append(("backbone.stem.conv1.weight", "model.pixel_level_module.encoder.embedder.embedder.convolution.weight"))
    rename_keys.append(("backbone.stem.conv1.norm.weight", "model.pixel_level_module.encoder.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.stem.conv1.norm.bias", "model.pixel_level_module.encoder.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.stem.conv1.norm.running_mean", "model.pixel_level_module.encoder.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.stem.conv1.norm.running_var", "model.pixel_level_module.encoder.embedder.embedder.normalization.running_var"))
    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            if layer_idx == 0:
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.weight",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.norm.weight",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.norm.bias",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.norm.running_mean",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.norm.running_var",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_var",
                    )
                )
            for i in range(3):
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i + 1}.weight",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i + 1}.norm.weight",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i + 1}.norm.bias",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i + 1}.norm.running_mean",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i + 1}.norm.running_var",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.running_var",
                    )
                )

    rename_keys.append(("sem_seg_head.layer_4.weight", "model.pixel_level_module.decoder.fpn.stem.0.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.weight", "model.pixel_level_module.decoder.fpn.stem.1.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.bias", "model.pixel_level_module.decoder.fpn.stem.1.bias"))
    for source_index, target_index in zip(range(3, 0, -1), range(0, 3)):
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.0.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.bias"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.0.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.bias"))
    rename_keys.append(("sem_seg_head.mask_features.weight", "model.pixel_level_module.decoder.mask_projection.weight"))
    rename_keys.append(("sem_seg_head.mask_features.bias", "model.pixel_level_module.decoder.mask_projection.bias"))

    for idx in range(config.decoder_config.decoder_layers):
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.weight", f"model.transformer_module.decoder.layers.{idx}.fc1.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.bias", f"model.transformer_module.decoder.layers.{idx}.fc1.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.weight", f"model.transformer_module.decoder.layers.{idx}.fc2.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.bias", f"model.transformer_module.decoder.layers.{idx}.fc2.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.weight", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.bias", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.bias"))

    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.weight", "model.transformer_module.decoder.layernorm.weight"))
    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.bias", "model.transformer_module.decoder.layernorm.bias"))

    rename_keys.append(("sem_seg_head.predictor.query_embed.weight", "model.transformer_module.queries_embedder.weight"))

    rename_keys.append(("sem_seg_head.predictor.input_proj.weight", "model.transformer_module.input_projection.weight"))
    rename_keys.append(("sem_seg_head.predictor.input_proj.bias", "model.transformer_module.input_projection.bias"))

    rename_keys.append(("sem_seg_head.predictor.class_embed.weight", "class_predictor.weight"))
    rename_keys.append(("sem_seg_head.predictor.class_embed.bias", "class_predictor.bias"))

    for i in range(3):
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.weight", f"mask_embedder.{i}.0.weight"))
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.bias", f"mask_embedder.{i}.0.bias"))

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def read_in_decoder_q_k_v(state_dict, config):
    pass


def prepare_img() -> torch.Tensor:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_maskformer_checkpoint(
    model_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="maskformer-resnet50-ade",
        type=str,
        required=True,
        choices=[
            "maskformer-resnet50-ade",
            "maskformer-resnet101-ade",
            "maskformer-resnet50-coco-stuff",
            "maskformer-resnet101-coco-stuff",
            "maskformer-resnet101-cityscapes",
            "maskformer-resnet50-vistas",
            "maskformer-resnet50-ade20k-full",
            "maskformer-resnet101-ade20k-full",
        ],
        help=("Name of the MaskFormer model you'd like to convert",),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the original pickle file (.pkl) of the original checkpoint.\n"
        "Given the files are in the pickle format, please be wary of passing it files you trust.",
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
    convert_maskformer_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
