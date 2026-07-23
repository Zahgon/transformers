
import argparse
import collections
import os
import re
from io import BytesIO

import httpx
import numpy as np
import torch
from PIL import Image

from transformers import CLIPImageProcessor

from ...utils import logging
from .configuration_mlcd import MLCDVisionConfig
from .modeling_mlcd import MLCDVisionModel


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


COMMON_CONFIG_PARAMS = {
    "mlcd-vit-bigG-patch14-336": {
        "hidden_size": 1664,
        "image_size": 336,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 48,
        "patch_size": 14,
        "projection_dim": 1024,
    },
    "mlcd-vit-bigG-patch14-448": {
        "hidden_size": 1664,
        "image_size": 448,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 48,
        "patch_size": 14,
        "projection_dim": 1024,
    },
}

MODEL_NAME_TO_CHECKPOINT_PATH = {
    "mlcd-vit-bigG-patch14-336": "MLCD_ViT_bigG_14_336px_pytorch.pt",
    "mlcd-vit-bigG-patch14-448": "MLCD_ViT_bigG_14_448px_pytorch.pt",
}

EXPECTED_OUTPUTS = {
    "mlcd-vit-bigG-patch14-336": torch.tensor([
        [-0.8921, -0.1069,  0.2989,  0.6018, -0.5892],
        [ 0.4093, -1.4592,  0.6048, -0.5147, -0.5929],
        [ 0.7796, -0.7133, -0.5649, -0.7843, -0.5548],
        [ 0.0041,  0.0286,  0.4310, -0.1403, -0.2399],
        [ 0.0839,  0.2152, -0.3822, -0.1668, -0.7886]
    ]),
    "mlcd-vit-bigG-patch14-448": torch.tensor([
        [-0.8978, -0.1181,  0.4769,  0.4761, -0.5779],
        [ 0.2640, -2.6150,  0.4853,  0.5743, -1.1003],
        [ 0.3314, -0.3328, -0.4305, -0.1874, -0.7701],
        [-1.5174, -1.0238, -1.1854,  0.1749, -0.8786],
        [ 0.2323, -0.8346, -0.9680, -0.2951,  0.0867],
    ]),
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"conv1.weight":                                                r"vision_model.embeddings.patch_embedding.weight",
    r"class_embedding":                                             r"vision_model.embeddings.class_embedding",
    r"vision_rotary_embedding":                                     r"vision_model.vision_rotary_embedding",
    r"class_pos_emb":                                               r"vision_model.class_pos_emb",
    r"transformer.resblocks_(\d+).ln_1.weight":                     r"vision_model.encoder.layers.\1.layer_norm1.weight",
    r"transformer.resblocks_(\d+).ln_1.bias":                       r"vision_model.encoder.layers.\1.layer_norm1.bias",
    r"transformer.resblocks_(\d+).ln_2.weight":                     r"vision_model.encoder.layers.\1.layer_norm2.weight",
    r"transformer.resblocks_(\d+).ln_2.bias":                       r"vision_model.encoder.layers.\1.layer_norm2.bias",
    r"transformer.resblocks_(\d+).mlp.c_fc.weight":                 r"vision_model.encoder.layers.\1.mlp.fc1.weight",
    r"transformer.resblocks_(\d+).mlp.c_fc.bias":                   r"vision_model.encoder.layers.\1.mlp.fc1.bias",
    r"transformer.resblocks_(\d+).mlp.c_proj.weight":               r"vision_model.encoder.layers.\1.mlp.fc2.weight",
    r"transformer.resblocks_(\d+).mlp.c_proj.bias":                 r"vision_model.encoder.layers.\1.mlp.fc2.bias",
    r"transformer.resblocks_(\d+).attn.(q|k|v|out)_proj.weight":    r"vision_model.encoder.layers.\1.self_attn.\2_proj.weight",
    r"transformer.resblocks_(\d+).attn.(q|k|v|out)_proj.bias":      r"vision_model.encoder.layers.\1.self_attn.\2_proj.bias",
    r"ln_post.weight":                                              r"vision_model.post_layernorm.weight",
    r"ln_post.bias":                                                r"vision_model.post_layernorm.bias",
    r"ln_pre.weight":                                               r"vision_model.pre_layernorm.weight",
    r"ln_pre.bias":                                                 r"vision_model.pre_layernorm.bias",
}




def get_mlcd_config(model_name: str) -> MLCDVisionConfig:
    pass


def get_mlcd_image_processor(model_name: str) -> CLIPImageProcessor:
    pass




def flatten_nested_dict(params: dict, parent_key: str = "", sep: str = ".") -> dict:
    pass


def split_resblocks_layers(state_dict: dict) -> dict:
    pass


def chunk_qkv_for_attn(state_dict: dict) -> dict:
    pass


def convert_old_keys_to_new_keys(state_dict_keys: list) -> dict:
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict




@torch.no_grad()
def convert_mlcd_checkpoint(model_name, input_dir, output_dir, verify_hidden_state=True, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="mlcd-vit-bigG-patch14-448",
        type=str,
        choices=MODEL_NAME_TO_CHECKPOINT_PATH.keys(),
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--input_dir",
        default="mlcd/original",
        help="Location of MLCD original weights",
    )
    parser.add_argument(
        "--output_dir",
        default="mlcd/checkpoint",
        help="Location to write HF model and processor",
    )
    parser.add_argument(
        "--verify_hidden_state",
        action="store_true",
        help="Whether to verify hidden_state against the original implementation.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_mlcd_checkpoint(
        args.model_name, args.input_dir, args.output_dir, args.verify_hidden_state, args.push_to_hub
    )
