
import argparse
from collections.abc import MutableMapping

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from clip.model import CLIP
from flax.training import checkpoints
from huggingface_hub import Repository

from transformers import (
    CLIPTokenizer,
    OwlViTConfig,
    OwlViTForObjectDetection,
    OwlViTImageProcessor,
    OwlViTModel,
    OwlViTProcessor,
)


CONFIGS = {
    "vit_b32": {
        "embed_dim": 512,
        "image_resolution": 768,
        "context_length": 16,
        "vocab_size": 49408,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 32,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
    },
    "vit_b16": {
        "embed_dim": 512,
        "image_resolution": 768,
        "context_length": 16,
        "vocab_size": 49408,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 16,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
    },
    "vit_l14": {
        "embed_dim": 768,
        "image_resolution": 840,
        "context_length": 16,
        "vocab_size": 49408,
        "vision_layers": 24,
        "vision_width": 1024,
        "vision_patch_size": 14,
        "transformer_width": 768,
        "transformer_heads": 12,
        "transformer_layers": 12,
    },
}


def flatten_nested_dict(params, parent_key="", sep="/"):
    pass


def to_f32(params):
    pass


def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    pass


def copy_mlp(hf_mlp, pt_mlp):
    pass


def copy_linear(hf_linear, pt_linear):
    pass


def copy_layer(hf_layer, pt_layer):
    pass


def copy_layers(hf_layers, pt_layers):
    pass


def copy_encoder(hf_encoder, pt_model):
    pass


def copy_text_model_and_projection(hf_model, pt_model):
    pass


def copy_vision_model_and_projection(hf_model, pt_model):
    pass


def copy_class_merge_token(hf_model, flax_params):
    pass


def copy_class_box_heads(hf_model, flax_params):
    pass


def copy_flax_attn_params(hf_backbone, flax_attn_params):
    pass


def _convert_attn_layers(params):
    pass


def convert_clip_backbone(flax_params, torch_config):
    pass


@torch.no_grad()
def convert_owlvit_checkpoint(pt_backbone, flax_params, attn_params, pytorch_dump_folder_path, config_path=None):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--owlvit_version",
        default=None,
        type=str,
        required=True,
        help="OWL-ViT model name [clip_b16, clip_b32, clip_l14].",
    )
    parser.add_argument(
        "--owlvit_checkpoint", default=None, type=str, required=True, help="Path to flax model checkpoint."
    )
    parser.add_argument("--hf_config", default=None, type=str, required=True, help="Path to HF model config.")
    parser.add_argument(
        "--pytorch_dump_folder_path", default="hf_model", type=str, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()

    model_name = args.owlvit_version
    if model_name == "clip_b16":
        torch_config = CONFIGS["vit_b16"]
    elif model_name == "clip_b32":
        torch_config = CONFIGS["vit_b32"]
    elif model_name == "clip_l14":
        torch_config = CONFIGS["vit_l14"]

    variables = checkpoints.restore_checkpoint(args.owlvit_checkpoint, target=None)["optimizer"]["target"]
    flax_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, variables)
    del variables

    pt_backbone_params, clip_pt, attn_params = convert_clip_backbone(flax_params, torch_config)

    convert_owlvit_checkpoint(clip_pt, flax_params, attn_params, args.pytorch_dump_folder_path, args.hf_config)
