
import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import functional as F

from transformers import DetrImageProcessor, ResNetConfig, TableTransformerConfig, TableTransformerForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def create_rename_keys(config):
    rename_keys = []

    rename_keys.append(("backbone.0.body.conv1.weight", "backbone.conv_encoder.model.embedder.embedder.convolution.weight"))
    rename_keys.append(("backbone.0.body.bn1.weight", "backbone.conv_encoder.model.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.0.body.bn1.bias", "backbone.conv_encoder.model.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.0.body.bn1.running_mean", "backbone.conv_encoder.model.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.0.body.bn1.running_var", "backbone.conv_encoder.model.embedder.embedder.normalization.running_var"))
    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.conv1.weight",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.convolution.weight",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn1.weight",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.normalization.weight",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn1.bias",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.normalization.bias",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn1.running_mean",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.normalization.running_mean",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn1.running_var",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.normalization.running_var",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.conv2.weight",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.convolution.weight",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn2.weight",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.normalization.weight",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn2.bias",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.normalization.bias",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn2.running_mean",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.normalization.running_mean",
                )
            )
            rename_keys.append(
                (
                    f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.bn2.running_var",
                    f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.normalization.running_var",
                )
            )
            if stage_idx != 0 and layer_idx == 0:
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.0.weight",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.weight",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.bias",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.running_mean",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.0.body.layer{stage_idx + 1}.{layer_idx}.downsample.1.running_var",
                        f"backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_var",
                    )
                )

    for i in range(config.encoder_layers):
        rename_keys.append(
            (
                f"transformer.encoder.layers.{i}.self_attn.out_proj.weight",
                f"encoder.layers.{i}.self_attn.out_proj.weight",
            )
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
        rename_keys.append(
            (f"transformer.encoder.layers.{i}.norm1.bias", f"encoder.layers.{i}.self_attn_layer_norm.bias")
        )
        rename_keys.append(
            (f"transformer.encoder.layers.{i}.norm2.weight", f"encoder.layers.{i}.final_layer_norm.weight")
        )
        rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"encoder.layers.{i}.final_layer_norm.bias"))
        rename_keys.append(
            (
                f"transformer.decoder.layers.{i}.self_attn.out_proj.weight",
                f"decoder.layers.{i}.self_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.out_proj.bias")
        )
        rename_keys.append(
            (
                f"transformer.decoder.layers.{i}.multihead_attn.out_proj.weight",
                f"decoder.layers.{i}.encoder_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"transformer.decoder.layers.{i}.multihead_attn.out_proj.bias",
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
        rename_keys.append(
            (f"transformer.decoder.layers.{i}.norm1.bias", f"decoder.layers.{i}.self_attn_layer_norm.bias")
        )
        rename_keys.append(
            (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.encoder_attn_layer_norm.weight")
        )
        rename_keys.append(
            (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.encoder_attn_layer_norm.bias")
        )
        rename_keys.append(
            (f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight")
        )
        rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))

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
            ("transformer.encoder.norm.weight", "encoder.layernorm.weight"),
            ("transformer.encoder.norm.bias", "encoder.layernorm.bias"),
        ]
    )

    return rename_keys


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""
    if is_panoptic:
        prefix = "detr."

    for i in range(6):
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
    for i in range(6):
        in_proj_weight = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
        in_proj_weight_cross_attn = state_dict.pop(
            f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_weight"
        )
        in_proj_bias_cross_attn = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_bias")
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]


def resize(image, checkpoint_url):
    width, height = image.size
    current_max_size = max(width, height)
    target_max_size = 800 if "detection" in checkpoint_url else 1000
    scale = target_max_size / current_max_size
    resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

    return resized_image


def normalize(image):
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image


@torch.no_grad()
def convert_table_transformer_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://pubtables1m.blob.core.windows.net/model/pubtables1m_detection_detr_r18.pth",
        type=str,
        choices=[
            "https://pubtables1m.blob.core.windows.net/model/pubtables1m_detection_detr_r18.pth",
            "https://pubtables1m.blob.core.windows.net/model/pubtables1m_structure_detr_r18.pth",
        ],
        help="URL of the Table Transformer checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )
    args = parser.parse_args()
    convert_table_transformer_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
