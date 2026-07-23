import argparse
import json
import os
import re

import torch
from safetensors.torch import load_file

from transformers import DogeConfig, DogeForCausalLM


STATE_DICT_MAPPING = {
    r"^lm_head.weight": r"lm_head.weight",

    r"^model.word_embed.weight": r"model.embed_tokens.weight",
    r"^model.rotary_emb.rotary_emb": r"model.rotary_emb.rotary_emb",
    r"^model.final_layernorm.weight": r"model.norm.weight",

    r"^model.layers.(\d+).pre_layernorm.weight": r"model.layers.\1.input_layernorm.weight",
    r"^model.layers.(\d+).pre_residual.weight": r"model.layers.\1.input_residual",
    r"^model.layers.(\d+).post_layernorm.weight": r"model.layers.\1.post_attention_layernorm.weight",
    r"^model.layers.(\d+).post_residual.weight": r"model.layers.\1.post_attention_residual",

    r"^model.layers.(\d+).self_attn.q_proj.weight": r"model.layers.\1.self_attn.q_proj.weight",
    r"^model.layers.(\d+).self_attn.k_proj.weight": r"model.layers.\1.self_attn.k_proj.weight",
    r"^model.layers.(\d+).self_attn.v_proj.weight": r"model.layers.\1.self_attn.v_proj.weight",
    r"^model.layers.(\d+).self_attn.A": r"model.layers.\1.self_attn.A",
    r"^model.layers.(\d+).self_attn.dt_proj.weight": r"model.layers.\1.self_attn.dt_proj.weight",
    r"^model.layers.(\d+).self_attn.o_proj.weight": r"model.layers.\1.self_attn.o_proj.weight",

    r"^model.layers.(\d+).feed_forward.gate_proj.weight": r"model.layers.\1.mlp.gate_proj.weight",
    r"^model.layers.(\d+).feed_forward.up_proj.weight": r"model.layers.\1.mlp.up_proj.weight",
    r"^model.layers.(\d+).feed_forward.down_proj.weight": r"model.layers.\1.mlp.down_proj.weight",
    r"^model.layers.(\d+).feed_forward.router_gate.weight": r"model.layers.\1.mlp.router_gate.weight",
    r"^model.layers.(\d+).feed_forward.router_gate.bias": None,
    r"^model.layers.(\d+).feed_forward.down_embed.weight": r"model.layers.\1.mlp.down_embed.weight",
    r"^model.layers.(\d+).feed_forward.up_embed.weight": r"model.layers.\1.mlp.up_embed.weight",
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


def convert_state_dict(original_state_dict: dict, config: DogeConfig):
    new_dict = {}

    for old_key, value in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        if new_key is None:
            continue
        new_dict[new_key] = value
    return new_dict


def convert_doge_model(input_dir, output_dir):
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
        help="Location to write HF model.",
    )

    args = parser.parse_args()
    convert_doge_model(args.input_dir, args.output_dir)
