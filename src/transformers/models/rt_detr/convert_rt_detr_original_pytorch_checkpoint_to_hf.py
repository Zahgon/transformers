
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_rt_detr_config(model_name: str) -> RTDetrConfig:
    pass


def create_rename_keys(config):
    rename_keys = []

    last_key = ["weight", "bias", "running_mean", "running_var"]

    for level in range(3):
        rename_keys.append((f"backbone.conv1.conv1_{level+1}.conv.weight", f"model.backbone.model.embedder.embedder.{level}.convolution.weight"))
        for last in last_key:
            rename_keys.append((f"backbone.conv1.conv1_{level+1}.norm.{last}", f"model.backbone.model.embedder.embedder.{level}.normalization.{last}"))

    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            if layer_idx == 0:
                if stage_idx == 0:
                    rename_keys.append(
                        (
                            f"backbone.res_layers.{stage_idx}.blocks.0.short.conv.weight",
                            f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.convolution.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append(
                            (
                                f"backbone.res_layers.{stage_idx}.blocks.0.short.norm.{last}",
                                f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.normalization.{last}",
                            )
                        )
                else:
                    rename_keys.append(
                        (
                            f"backbone.res_layers.{stage_idx}.blocks.0.short.conv.conv.weight",
                            f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.1.convolution.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append(
                            (
                                f"backbone.res_layers.{stage_idx}.blocks.0.short.conv.norm.{last}",
                                f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.1.normalization.{last}",
                            )
                        )

            rename_keys.append(
                (
                    f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2a.conv.weight",
                    f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.convolution.weight",
                )
            )
            for last in last_key:
                rename_keys.append((
                    f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2a.norm.{last}",
                    f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.normalization.{last}",
                    ))

            rename_keys.append(
                (
                    f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2b.conv.weight",
                    f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.convolution.weight",
                )
            )
            for last in last_key:
                rename_keys.append((
                    f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2b.norm.{last}",
                    f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.normalization.{last}",
                    ))

            if config.backbone_config.layer_type != "basic":
                rename_keys.append(
                    (
                        f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2c.conv.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.2.convolution.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append((
                        f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2c.norm.{last}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.2.normalization.{last}",
                        ))

    for i in range(config.encoder_layers):
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.self_attn.out_proj.weight",
                f"model.encoder.encoder.{i}.layers.0.self_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.self_attn.out_proj.bias",
                f"model.encoder.encoder.{i}.layers.0.self_attn.out_proj.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear1.weight",
                f"model.encoder.encoder.{i}.layers.0.fc1.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear1.bias",
                f"model.encoder.encoder.{i}.layers.0.fc1.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear2.weight",
                f"model.encoder.encoder.{i}.layers.0.fc2.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear2.bias",
                f"model.encoder.encoder.{i}.layers.0.fc2.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm1.weight",
                f"model.encoder.encoder.{i}.layers.0.self_attn_layer_norm.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm1.bias",
                f"model.encoder.encoder.{i}.layers.0.self_attn_layer_norm.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm2.weight",
                f"model.encoder.encoder.{i}.layers.0.final_layer_norm.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm2.bias",
                f"model.encoder.encoder.{i}.layers.0.final_layer_norm.bias",
            )
        )

    for j in range(0, 3):
        rename_keys.append((f"encoder.input_proj.{j}.0.weight", f"model.encoder_input_proj.{j}.0.weight"))
        for last in last_key:
            rename_keys.append((f"encoder.input_proj.{j}.1.{last}", f"model.encoder_input_proj.{j}.1.{last}"))

    block_levels = 3 if config.backbone_config.layer_type != "basic" else 4

    for i in range(len(config.encoder_in_channels) - 1):
        for j in range(1, block_levels):
            rename_keys.append(
                (f"encoder.fpn_blocks.{i}.conv{j}.conv.weight", f"model.encoder.fpn_blocks.{i}.conv{j}.conv.weight")
            )
            for last in last_key:
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.conv{j}.norm.{last}",
                        f"model.encoder.fpn_blocks.{i}.conv{j}.norm.{last}",
                    )
                )

        rename_keys.append((f"encoder.lateral_convs.{i}.conv.weight", f"model.encoder.lateral_convs.{i}.conv.weight"))
        for last in last_key:
            rename_keys.append(
                (f"encoder.lateral_convs.{i}.norm.{last}", f"model.encoder.lateral_convs.{i}.norm.{last}")
            )

        for j in range(3):
            for k in range(1, 3):
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                        f"model.encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.norm.{last}",
                            f"model.encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.norm.{last}",
                        )
                    )

        for j in range(1, block_levels):
            rename_keys.append(
                (f"encoder.pan_blocks.{i}.conv{j}.conv.weight", f"model.encoder.pan_blocks.{i}.conv{j}.conv.weight")
            )
            for last in last_key:
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.conv{j}.norm.{last}",
                        f"model.encoder.pan_blocks.{i}.conv{j}.norm.{last}",
                    )
                )

        for j in range(3):
            for k in range(1, 3):
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                        f"model.encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.norm.{last}",
                            f"model.encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.norm.{last}",
                        )
                    )

        rename_keys.append(
            (f"encoder.downsample_convs.{i}.conv.weight", f"model.encoder.downsample_convs.{i}.conv.weight")
        )
        for last in last_key:
            rename_keys.append(
                (f"encoder.downsample_convs.{i}.norm.{last}", f"model.encoder.downsample_convs.{i}.norm.{last}")
            )

    for i in range(config.decoder_layers):
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.self_attn.out_proj.weight",
                f"model.decoder.layers.{i}.self_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.self_attn.out_proj.bias",
                f"model.decoder.layers.{i}.self_attn.out_proj.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.sampling_offsets.weight",
                f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.sampling_offsets.bias",
                f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.attention_weights.weight",
                f"model.decoder.layers.{i}.encoder_attn.attention_weights.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.attention_weights.bias",
                f"model.decoder.layers.{i}.encoder_attn.attention_weights.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.value_proj.weight",
                f"model.decoder.layers.{i}.encoder_attn.value_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.value_proj.bias",
                f"model.decoder.layers.{i}.encoder_attn.value_proj.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.output_proj.weight",
                f"model.decoder.layers.{i}.encoder_attn.output_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.output_proj.bias",
                f"model.decoder.layers.{i}.encoder_attn.output_proj.bias",
            )
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm1.weight", f"model.decoder.layers.{i}.self_attn_layer_norm.weight")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm1.bias", f"model.decoder.layers.{i}.self_attn_layer_norm.bias")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm2.weight", f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm2.bias", f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias")
        )
        rename_keys.append((f"decoder.decoder.layers.{i}.linear1.weight", f"model.decoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear1.bias", f"model.decoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear2.weight", f"model.decoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear2.bias", f"model.decoder.layers.{i}.fc2.bias"))
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm3.weight", f"model.decoder.layers.{i}.final_layer_norm.weight")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm3.bias", f"model.decoder.layers.{i}.final_layer_norm.bias")
        )

    for i in range(config.decoder_layers):
        rename_keys.append(
            (
                f"decoder.dec_score_head.{i}.weight",
                f"model.decoder.class_embed.{i}.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_score_head.{i}.bias",
                f"model.decoder.class_embed.{i}.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.0.weight",
                f"model.decoder.bbox_embed.{i}.layers.0.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.0.bias",
                f"model.decoder.bbox_embed.{i}.layers.0.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.1.weight",
                f"model.decoder.bbox_embed.{i}.layers.1.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.1.bias",
                f"model.decoder.bbox_embed.{i}.layers.1.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.2.weight",
                f"model.decoder.bbox_embed.{i}.layers.2.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.2.bias",
                f"model.decoder.bbox_embed.{i}.layers.2.bias",
            )
        )

    for i in range(len(config.decoder_in_channels)):
        rename_keys.append(
            (
                f"decoder.input_proj.{i}.conv.weight",
                f"model.decoder_input_proj.{i}.0.weight",
            )
        )
        for last in last_key:
            rename_keys.append(
                (
                    f"decoder.input_proj.{i}.norm.{last}",
                    f"model.decoder_input_proj.{i}.1.{last}",
                )
            )

    rename_keys.extend(
        [
            ("decoder.denoising_class_embed.weight", "model.denoising_class_embed.weight"),
            ("decoder.query_pos_head.layers.0.weight", "model.decoder.query_pos_head.layers.0.weight"),
            ("decoder.query_pos_head.layers.0.bias", "model.decoder.query_pos_head.layers.0.bias"),
            ("decoder.query_pos_head.layers.1.weight", "model.decoder.query_pos_head.layers.1.weight"),
            ("decoder.query_pos_head.layers.1.bias", "model.decoder.query_pos_head.layers.1.bias"),
            ("decoder.enc_output.0.weight", "model.enc_output.0.weight"),
            ("decoder.enc_output.0.bias", "model.enc_output.0.bias"),
            ("decoder.enc_output.1.weight", "model.enc_output.1.weight"),
            ("decoder.enc_output.1.bias", "model.enc_output.1.bias"),
            ("decoder.enc_score_head.weight", "model.enc_score_head.weight"),
            ("decoder.enc_score_head.bias", "model.enc_score_head.bias"),
            ("decoder.enc_bbox_head.layers.0.weight", "model.enc_bbox_head.layers.0.weight"),
            ("decoder.enc_bbox_head.layers.0.bias", "model.enc_bbox_head.layers.0.bias"),
            ("decoder.enc_bbox_head.layers.1.weight", "model.enc_bbox_head.layers.1.weight"),
            ("decoder.enc_bbox_head.layers.1.bias", "model.enc_bbox_head.layers.1.bias"),
            ("decoder.enc_bbox_head.layers.2.weight", "model.enc_bbox_head.layers.2.weight"),
            ("decoder.enc_bbox_head.layers.2.bias", "model.enc_bbox_head.layers.2.bias"),
        ]
    )

    return rename_keys


def rename_key(state_dict, old, new):
    if old in state_dict:
        state_dict[new] = state_dict.pop(old)


def read_in_q_k_v(state_dict, config):
    prefix = ""
    encoder_hidden_dim = config.encoder_hidden_dim

    for i in range(config.encoder_layers):
        in_proj_weight = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_bias")
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.weight"] = in_proj_weight[
            :encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.bias"] = in_proj_bias[:encoder_hidden_dim]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.weight"] = in_proj_weight[
            encoder_hidden_dim : 2 * encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.bias"] = in_proj_bias[
            encoder_hidden_dim : 2 * encoder_hidden_dim
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.weight"] = in_proj_weight[
            -encoder_hidden_dim:, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.bias"] = in_proj_bias[-encoder_hidden_dim:]
    for i in range(config.decoder_layers):
        in_proj_weight = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_bias")
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_rt_detr_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, repo_id):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="rtdetr_r50vd",
        type=str,
        help="model_name of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="repo_id where the model will be pushed to.",
    )
    args = parser.parse_args()
    convert_rt_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
