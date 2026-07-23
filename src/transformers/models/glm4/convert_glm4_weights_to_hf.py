import argparse
import json
import os
import re

import torch
from safetensors.torch import load_file
from tokenizers import processors

from transformers import Glm4Config, Glm4ForCausalLM, PreTrainedTokenizerFast


STATE_DICT_MAPPING = {
    r"transformer.output_layer.weight":                                               r"lm_head.weight",

    r"transformer.embedding.word_embeddings.weight":                                  r"model.embed_tokens.weight",
    r"transformer.rotary_pos_emb.inv_freq":                                           None,
    r"transformer.encoder.final_layernorm.weight":                                    r"model.norm.weight",

    r"transformer.encoder.layers.(\d+).input_layernorm.weight":                       r"model.layers.\1.input_layernorm.weight",

    r"transformer.encoder.layers.(\d+).post_mlp_layernorm.weight":                    r"model.layers.\1.post_mlp_layernorm.weight",
    r"transformer.encoder.layers.(\d+).post_self_attn_layernorm.weight":              r"model.layers.\1.post_self_attn_layernorm.weight",

    r"transformer.encoder.layers.(\d+).post_attention_layernorm.weight":              r"model.layers.\1.post_attention_layernorm.weight",

    r"transformer.encoder.layers.(\d+).self_attention.dense.weight":                  r"model.layers.\1.self_attn.o_proj.weight",
    r"transformer.encoder.layers.(\d+).self_attention.query_key_value.(weight|bias)": r"model.layers.\1.self_attn.qkv_proj.\2",

    r"transformer.encoder.layers.(\d+).mlp.dense_h_to_4h.weight":                     r"model.layers.\1.mlp.gate_up_proj.weight",
    r"transformer.encoder.layers.(\d+).mlp.dense_4h_to_h.weight":                     r"model.layers.\1.mlp.down_proj.weight",
}


def load_weights(input_dir: str):
    pass


def map_old_key_to_new(old_key):
    for pattern, replacement in STATE_DICT_MAPPING.items():
        if replacement is None:
            if re.fullmatch(pattern, old_key):
                return None
        else:
            new_key, n_replace = re.subn(pattern, replacement, old_key)
            if n_replace > 0:
                return new_key

    raise ValueError(f"Key: {old_key} could not be mapped (check the mapping).")


def convert_state_dict(original_state_dict: dict, config: Glm4Config):
    new_dict = {}

    head_dim = config.hidden_size // config.num_attention_heads
    query_size = config.num_attention_heads * head_dim
    kv_size = config.num_key_value_heads * head_dim

    for old_key, value in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        if new_key is None:
            continue

        if "qkv_proj." in new_key:
            q_proj, k_proj, v_proj = (
                value[:query_size, ...],
                value[query_size : query_size + kv_size, ...],
                value[query_size + kv_size :, ...],
            )
            new_dict[new_key.replace("qkv_proj.", "q_proj.")] = q_proj
            new_dict[new_key.replace("qkv_proj.", "k_proj.")] = k_proj
            new_dict[new_key.replace("qkv_proj.", "v_proj.")] = v_proj
        else:
            new_dict[new_key] = value
    return new_dict


def convert_config(original_config: dict):
    key_mapping = {
        "vocab_size": "padded_vocab_size",
        "intermediate_size": "ffn_hidden_size",
        "num_hidden_layers": "num_layers",
        "max_position_embeddings": "seq_length",
        "rms_norm_eps": "layernorm_epsilon",
        "head_dim": "kv_channels",
        "attention_bias": "add_qkv_bias",
    }
    similar_keys_to_keep = [
        "num_attention_heads",
        "hidden_size",
        "attention_dropout",
        "use_cache",
        "eos_token_id",
        "pad_token_id",
        "tie_word_embeddings",
    ]
    new_config_kwargs = {k: original_config[v] for k, v in key_mapping.items()}
    new_config_kwargs.update({k: v for k, v in original_config.items() if k in similar_keys_to_keep})
    new_config_kwargs["num_key_value_heads"] = (
        new_config_kwargs["num_attention_heads"]
        if not original_config["multi_query_attention"]
        else original_config["multi_query_group_num"]
    )
    new_config_kwargs["rope_theta"] = 10000.0 * getattr(original_config, "rope_ratio", 1)

    new_config = Glm4Config(**new_config_kwargs)
    return new_config


def convert_glm4_tokenizer(input_dir, use_post_processor=False):
    pass


def convert_glm4_model(input_dir, output_dir, use_post_processor=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Location of the local folder copied from the Hub.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--use_post_processor",
        action="store_true",
        help="Whether to apply post processor with special tokens",
    )
    args = parser.parse_args()
    convert_glm4_model(args.input_dir, args.output_dir, args.use_post_processor)
