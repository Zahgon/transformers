
import argparse
import collections
import os
import re

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

from transformers import Siglip2Config, Siglip2ImageProcessorFast, Siglip2Model, Siglip2Processor, Siglip2Tokenizer
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


COMMON_CONFIG_PARAMS = {
    "base": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
    },
    "large": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
    },
    "so400m": {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
    },
}

MODEL_NAME_TO_CHECKPOINT_PATH = {
    "siglip2-base-patch16-naflex": "gv-hf/siglip2/siglip2_b16_naflex.npz",
    "siglip2-so400m-patch16-naflex": "gv-hf/siglip2/siglip2_so400m16_naflex.npz",
}

EXPECTED_OUTPUTS = {
    "siglip2-base-patch16-naflex": torch.tensor([
        [  1.0195,  -0.0280,  -1.4468],
        [ -4.5395,  -6.2269,  -1.5667],
        [  4.1757,   5.0358,   3.5159],
        [  9.4264,  10.1879,   6.3353],
        [  2.4409,   3.1058,   4.5491],
        [-12.3230, -13.7355, -13.4632],
        [  1.1520,   1.1687,  -1.9647],
    ]),
    "siglip2-so400m-patch16-naflex": torch.tensor([
        [  0.9422,   0.5540,  -2.4405],
        [ -7.3522,  -9.4931,  -6.3499],
        [  5.7852,   6.7288,   7.7893],
        [  9.9881,  10.8136,   9.2121],
        [  5.3660,   5.7746,   8.4130],
        [-12.7218, -14.2631, -13.6442],
        [  0.6384,   0.4278,  -0.9022],
    ]),
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"params/img/embedding/kernel":                                                                         r"vision_model.embeddings.patch_embedding.weight",
    r"params/img/embedding/bias":                                                                           r"vision_model.embeddings.patch_embedding.bias",
    r"params/img/pos_embedding":                                                                            r"vision_model.embeddings.position_embedding.weight",
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_0/scale":                                         r"vision_model.encoder.layers.\1.layer_norm1.weight",
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_0/bias":                                          r"vision_model.encoder.layers.\1.layer_norm1.bias",
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_1/scale":                                         r"vision_model.encoder.layers.\1.layer_norm2.weight",
    r"params/img/Transformer/encoderblock_(\d+)/LayerNorm_1/bias":                                          r"vision_model.encoder.layers.\1.layer_norm2.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_0/kernel":                                 r"vision_model.encoder.layers.\1.mlp.fc1.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_0/bias":                                   r"vision_model.encoder.layers.\1.mlp.fc1.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_1/kernel":                                 r"vision_model.encoder.layers.\1.mlp.fc2.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MlpBlock_0/Dense_1/bias":                                   r"vision_model.encoder.layers.\1.mlp.fc2.bias",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/(q|k|v|out)[a-z]*/kernel":   r"vision_model.encoder.layers.\1.self_attn.\2_proj.weight",
    r"params/img/Transformer/encoderblock_(\d+)/MultiHeadDotProductAttention_0/(q|k|v|out)[a-z]*/bias":     r"vision_model.encoder.layers.\1.self_attn.\2_proj.bias",
    r"params/img/Transformer/encoder_norm/scale":                                                           r"vision_model.post_layernorm.weight",
    r"params/img/Transformer/encoder_norm/bias":                                                            r"vision_model.post_layernorm.bias",
    r"params/img/MAPHead_0/probe":                                                                          r"vision_model.head.probe",
    r"params/img/MAPHead_0/LayerNorm_0/scale":                                                              r"vision_model.head.layernorm.weight",
    r"params/img/MAPHead_0/LayerNorm_0/bias":                                                               r"vision_model.head.layernorm.bias",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_0/kernel":                                                      r"vision_model.head.mlp.fc1.weight",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_0/bias":                                                        r"vision_model.head.mlp.fc1.bias",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_1/kernel":                                                      r"vision_model.head.mlp.fc2.weight",
    r"params/img/MAPHead_0/MlpBlock_0/Dense_1/bias":                                                        r"vision_model.head.mlp.fc2.bias",
    r"params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/kernel":                                      r"vision_model.head.attention.out_proj.weight",
    r"params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/bias":                                        r"vision_model.head.attention.out_proj.bias",
    r"params/img/MAPHead_0/MultiHeadDotProductAttention_0/qkv/kernel":                                      r"vision_model.head.attention.in_proj_weight",
    r"params/img/MAPHead_0/MultiHeadDotProductAttention_0/qkv/bias":                                        r"vision_model.head.attention.in_proj_bias",
    r"params/txt/Embed_0/embedding":                                                                        r"text_model.embeddings.token_embedding.weight",
    r"params/txt/pos_embedding":                                                                            r"text_model.embeddings.position_embedding.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_0/scale":                                           r"text_model.encoder.layers.\1.layer_norm1.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_0/bias":                                            r"text_model.encoder.layers.\1.layer_norm1.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_1/scale":                                           r"text_model.encoder.layers.\1.layer_norm2.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/LayerNorm_1/bias":                                            r"text_model.encoder.layers.\1.layer_norm2.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_0/kernel":                                   r"text_model.encoder.layers.\1.mlp.fc1.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_0/bias":                                     r"text_model.encoder.layers.\1.mlp.fc1.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_1/kernel":                                   r"text_model.encoder.layers.\1.mlp.fc2.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MlpBlock_0/Dense_1/bias":                                     r"text_model.encoder.layers.\1.mlp.fc2.bias",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/(q|k|v|out)[a-z]*/kernel":     r"text_model.encoder.layers.\1.self_attn.\2_proj.weight",
    r"params/txt/Encoder_0/encoderblock_(\d+)/MultiHeadDotProductAttention_0/(q|k|v|out)[a-z]*/bias":       r"text_model.encoder.layers.\1.self_attn.\2_proj.bias",
    r"params/txt/Encoder_0/encoder_norm/scale":                                                             r"text_model.final_layer_norm.weight",
    r"params/txt/Encoder_0/encoder_norm/bias":                                                              r"text_model.final_layer_norm.bias",
    r"params/txt/head/kernel":                                                                              r"text_model.head.weight",
    r"params/txt/head/bias":                                                                                r"text_model.head.bias",
    r"params/t":                                                                                            r"logit_scale",
    r"params/b":                                                                                            r"logit_bias",
}




def get_siglip2_config(model_name: str) -> Siglip2Config:
    pass


def get_siglip2_tokenizer() -> Siglip2Tokenizer:
    pass


def get_siglip2_image_processor(patch_size: int, max_num_patches: int) -> Siglip2ImageProcessorFast:
    pass




def flatten_nested_dict(params: dict, parent_key: str = "", sep: str = "/") -> dict:
    pass


def split_encoderblock_layers(state_dict: dict) -> dict:
    pass


def merge_qkv_for_head(state_dict: dict, config: Siglip2Config) -> dict:
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




def create_image(width, height):
    pass


def prepare_inputs():
    pass




@torch.no_grad()
def convert_siglip2_checkpoint(model_name, pytorch_dump_folder_path, verify_logits=True, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="siglip2-base-patch16-naflex",
        type=str,
        choices=MODEL_NAME_TO_CHECKPOINT_PATH.keys(),
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="checkpoints/",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--verify_logits",
        action="store_true",
        help="Whether to verify logits against the original implementation.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_siglip2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.verify_logits, args.push_to_hub)
