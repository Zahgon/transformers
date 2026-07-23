
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from transformers import AutoImageProcessor, CvtConfig, CvtForImageClassification


def embeddings(idx):
    """
    The function helps in renaming embedding layer weights.

    Args:
        idx: stage number in original model
    """
    embed = []
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.projection.weight",
            f"stage{idx}.patch_embed.proj.weight",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.projection.bias",
            f"stage{idx}.patch_embed.proj.bias",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.normalization.weight",
            f"stage{idx}.patch_embed.norm.weight",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.normalization.bias",
            f"stage{idx}.patch_embed.norm.bias",
        )
    )
    return embed


def attention(idx, cnt):
    """
    The function helps in renaming attention block layers weights.

    Args:
        idx: stage number in original model
        cnt: count of blocks in each stage
    """
    attention_weights = []
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.conv.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.bias",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.running_mean",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.running_mean",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.running_var",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.running_var",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.num_batches_tracked",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.num_batches_tracked",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.conv.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.bias",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.running_mean",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.running_mean",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.running_var",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.running_var",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.num_batches_tracked",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.num_batches_tracked",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.conv.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.bias",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.running_mean",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.running_mean",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.running_var",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.running_var",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.num_batches_tracked",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.num_batches_tracked",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_query.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj_q.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_query.bias",
            f"stage{idx}.blocks.{cnt}.attn.proj_q.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_key.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj_k.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_key.bias",
            f"stage{idx}.blocks.{cnt}.attn.proj_k.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_value.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj_v.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_value.bias",
            f"stage{idx}.blocks.{cnt}.attn.proj_v.bias",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.output.dense.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.output.dense.bias",
            f"stage{idx}.blocks.{cnt}.attn.proj.bias",
        )
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.intermediate.dense.weight", f"stage{idx}.blocks.{cnt}.mlp.fc1.weight")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.intermediate.dense.bias", f"stage{idx}.blocks.{cnt}.mlp.fc1.bias")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.output.dense.weight", f"stage{idx}.blocks.{cnt}.mlp.fc2.weight")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.output.dense.bias", f"stage{idx}.blocks.{cnt}.mlp.fc2.bias")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_before.weight", f"stage{idx}.blocks.{cnt}.norm1.weight")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_before.bias", f"stage{idx}.blocks.{cnt}.norm1.bias")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_after.weight", f"stage{idx}.blocks.{cnt}.norm2.weight")
    )
    attention_weights.append(
        (f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_after.bias", f"stage{idx}.blocks.{cnt}.norm2.bias")
    )
    return attention_weights


def cls_token(idx):
    pass


def final():
    """
    Function helps in renaming final classification layer
    """
    head = []
    head.append(("layernorm.weight", "norm.weight"))
    head.append(("layernorm.bias", "norm.bias"))
    head.append(("classifier.weight", "head.weight"))
    head.append(("classifier.bias", "head.bias"))
    return head


def convert_cvt_checkpoint(cvt_model, image_size, cvt_file_name, pytorch_dump_folder):
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cvt_model",
        default="cvt-w24",
        type=str,
        help="Name of the cvt model you'd like to convert.",
    )
    parser.add_argument(
        "--image_size",
        default=384,
        type=int,
        help="Input Image Size",
    )
    parser.add_argument(
        "--cvt_file_name",
        default=r"cvtmodels\CvT-w24-384x384-IN-22k.pth",
        type=str,
        help="Input Image Size",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_cvt_checkpoint(args.cvt_model, args.image_size, args.cvt_file_name, args.pytorch_dump_folder_path)
