
import argparse
import collections
import os
from io import BytesIO

import httpx
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from numpy import load
from PIL import Image

from transformers import (
    GemmaTokenizerFast,
    SiglipConfig,
    SiglipImageProcessor,
    SiglipModel,
    SiglipProcessor,
    SiglipTokenizer,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


MODEL_CONFIGS = {
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
    "giant-opt": {
        "hidden_size": 1536,
        "intermediate_size": 6144,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
    },
    "so400m": {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
    },
}

model_name_to_checkpoint = {
    "siglip-base-patch16-224": "https://storage.googleapis.com/big_vision/siglip/webli_en_b16_224_63724782.npz",
    "siglip-base-patch16-256": "https://storage.googleapis.com/big_vision/siglip/webli_en_b16_256_60500360.npz",
    "siglip-base-patch16-384": "https://storage.googleapis.com/big_vision/siglip/webli_en_b16_384_68578854.npz",
    "siglip-base-patch16-512": "https://storage.googleapis.com/big_vision/siglip/webli_en_b16_512_68580893.npz",
    "siglip-large-patch16-256": "https://storage.googleapis.com/big_vision/siglip/webli_en_l16_256_60552751.npz",
    "siglip-large-patch16-384": "https://storage.googleapis.com/big_vision/siglip/webli_en_l16_384_63634585.npz",
    "siglip-base-patch16-256-i18n": "https://storage.googleapis.com/big_vision/siglip/webli_i18n_b16_256_66117334.npz",
    "siglip-so400m-patch14-384": "https://storage.googleapis.com/big_vision/siglip/webli_en_so400m_384_58765454.npz",
    "siglip2-base-patch32-256": "gv-hf/siglip2/siglip2_b32_256.npz",
    "siglip2-base-patch16-224": "gv-hf/siglip2/siglip2_b16_224.npz",
    "siglip2-base-patch16-256": "gv-hf/siglip2/siglip2_b16_256.npz",
    "siglip2-base-patch16-384": "gv-hf/siglip2/siglip2_b16_384.npz",
    "siglip2-base-patch16-512": "gv-hf/siglip2/siglip2_b16_512.npz",
    "siglip2-large-patch16-256": "gv-hf/siglip2/siglip2_l16_256.npz",
    "siglip2-large-patch16-384": "gv-hf/siglip2/siglip2_l16_384.npz",
    "siglip2-large-patch16-512": "gv-hf/siglip2/siglip2_l16_512.npz",
    "siglip2-giant-opt-patch16-256": "gv-hf/siglip2/siglip2_g-opt16_256.npz",
    "siglip2-giant-opt-patch16-384": "gv-hf/siglip2/siglip2_g-opt16_384.npz",
    "siglip2-so400m-patch14-224": "gv-hf/siglip2/siglip2_so400m14_224.npz",
    "siglip2-so400m-patch14-384": "gv-hf/siglip2/siglip2_so400m14_384.npz",
    "siglip2-so400m-patch16-256": "gv-hf/siglip2/siglip2_so400m16_256.npz",
    "siglip2-so400m-patch16-384": "gv-hf/siglip2/siglip2_so400m16_384.npz",
    "siglip2-so400m-patch16-512": "gv-hf/siglip2/siglip2_so400m16_512.npz",
}



def get_image_size_from_model_name(model_name: str) -> int:
    pass


def get_patch_size_from_model_name(model_name: str) -> int:
    pass


def get_vocab_size_from_model_name(model_name: str) -> int:
    pass


def get_vocab_file_from_model_name(model_name: str) -> str:
    if "i18n" in model_name:
        repo_id = "google/siglip-base-patch16-256-multilingual"
    else:
        repo_id = "google/siglip-base-patch16-224"

    vocab_file = hf_hub_download(repo_id=repo_id, filename="spiece.model")
    return vocab_file


def get_text_and_vision_vit_variants(model_name: str) -> tuple[str, str]:
    pass


def get_siglip_config(model_name):
    pass




def get_tokenizer(model_name: str) -> GemmaTokenizerFast:
    if "siglip2" in model_name:
        tokenizer = GemmaTokenizerFast.from_pretrained(
            "google/gemma-2-9b-it",
            add_bos_token=False,
            add_eos_token=True,
            padding_side="right",
            do_lower_case=True,
            model_input_names=["input_ids"],
        )
    else:
        vocab_file = get_vocab_file_from_model_name(model_name)
        tokenizer = SiglipTokenizer(vocab_file=vocab_file, model_input_names=["input_ids"])
    return tokenizer


def get_image_processor(model_name: str) -> SiglipImageProcessor:
    pass




def split_encoderblock_layers(state_dict: dict) -> dict:
    pass


def create_rename_keys(config):
    rename_keys = []


    rename_keys.append(("params/img/embedding/kernel", "vision_model.embeddings.patch_embedding.weight"))
    rename_keys.append(("params/img/embedding/bias", "vision_model.embeddings.patch_embedding.bias"))
    rename_keys.append(("params/img/pos_embedding", "vision_model.embeddings.position_embedding.weight"))

    for i in range(config.vision_config.num_hidden_layers):
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_0/scale", f"vision_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_0/bias", f"vision_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_1/scale", f"vision_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_1/bias", f"vision_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_0/kernel", f"vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_0/bias", f"vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_1/kernel", f"vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_1/bias", f"vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/key/kernel", f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/key/bias", f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/value/kernel", f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/value/bias", f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/query/kernel", f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/query/bias", f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/out/kernel", f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/out/bias", f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))

    rename_keys.append(("params/img/Transformer/encoder_norm/scale", "vision_model.post_layernorm.weight"))
    rename_keys.append(("params/img/Transformer/encoder_norm/bias", "vision_model.post_layernorm.bias"))

    rename_keys.append(("params/img/MAPHead_0/probe", "vision_model.head.probe"))
    rename_keys.append(("params/img/MAPHead_0/LayerNorm_0/scale", "vision_model.head.layernorm.weight"))
    rename_keys.append(("params/img/MAPHead_0/LayerNorm_0/bias", "vision_model.head.layernorm.bias"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_0/kernel", "vision_model.head.mlp.fc1.weight"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_0/bias", "vision_model.head.mlp.fc1.bias"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_1/kernel", "vision_model.head.mlp.fc2.weight"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_1/bias", "vision_model.head.mlp.fc2.bias"))
    rename_keys.append(("params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/kernel", "vision_model.head.attention.out_proj.weight"))
    rename_keys.append(("params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/bias", "vision_model.head.attention.out_proj.bias"))


    rename_keys.append(("params/txt/Embed_0/embedding", "text_model.embeddings.token_embedding.weight"))
    rename_keys.append(("params/txt/pos_embedding", "text_model.embeddings.position_embedding.weight"))

    for i in range(config.text_config.num_hidden_layers):
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_0/scale", f"text_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_0/bias", f"text_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_1/scale", f"text_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_1/bias", f"text_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_0/kernel", f"text_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_0/bias", f"text_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_1/kernel", f"text_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_1/bias", f"text_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/key/kernel", f"text_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/key/bias", f"text_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/value/kernel", f"text_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/value/bias", f"text_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/query/kernel", f"text_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/query/bias", f"text_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/out/kernel", f"text_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/out/bias", f"text_model.encoder.layers.{i}.self_attn.out_proj.bias"))

    rename_keys.append(("params/txt/Encoder_0/encoder_norm/scale", "text_model.final_layer_norm.weight"))
    rename_keys.append(("params/txt/Encoder_0/encoder_norm/bias", "text_model.final_layer_norm.bias"))
    rename_keys.append(("params/txt/head/kernel", "text_model.head.weight"))
    rename_keys.append(("params/txt/head/bias", "text_model.head.bias"))

    rename_keys.append(("params/t", "logit_scale"))
    rename_keys.append(("params/b", "logit_bias"))

    return rename_keys


def rename_key(dct, old, new, config):
    val = dct.pop(old)

    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    if "patch_embedding.weight" in new:
        val = val.transpose(3, 2, 0, 1)
    elif new.endswith("weight") and "position_embedding" not in new and "token_embedding" not in new:
        val = val.T

    if "position_embedding" in new and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if "position_embedding" in new and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    if new.endswith("bias"):
        val = val.reshape(-1)

    dct[new] = torch.from_numpy(val)


def read_in_q_k_v_head(state_dict, config):
    pass


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


def flatten_nested_dict(params, parent_key="", sep="/"):
    pass


@torch.no_grad()
def convert_siglip_checkpoint(model_name, pytorch_dump_folder_path, verify_logits=True, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="siglip-base-patch16-224",
        type=str,
        choices=model_name_to_checkpoint.keys(),
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
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
    convert_siglip_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.verify_logits, args.push_to_hub)
