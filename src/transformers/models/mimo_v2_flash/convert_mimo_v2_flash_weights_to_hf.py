
import argparse
import json
import os

from transformers import MiMoV2FlashConfig


def convert_config(original_config: dict):
    keys_to_drop = {
        "attention_chunk_size",
        "sliding_window_size",
        "n_shared_experts",
        "scoring_func",
        "topk_method",
        "swa_num_attention_heads",
        "swa_num_key_value_heads",
        "swa_qk_head_dim",
        "swa_head_dim",
        "swa_v_head_dim",
        "add_swa_attention_sink_bias",
        "add_full_attention_sink_bias",
        "auto_map",
    }
    new_config_kwargs = {k: v for k, v in original_config.items() if k not in keys_to_drop}

    if "layernorm_epsilon" in new_config_kwargs:
        new_config_kwargs["rms_norm_eps"] = new_config_kwargs.pop("layernorm_epsilon")

    pattern = new_config_kwargs.pop("hybrid_layer_pattern", None)
    if pattern is not None:
        new_config_kwargs["layer_types"] = ["sliding_attention" if p == 1 else "full_attention" for p in pattern]
    freq = new_config_kwargs.pop("moe_layer_freq", None)
    if freq is not None:
        new_config_kwargs["mlp_layer_types"] = ["sparse" if f == 1 else "dense" for f in freq]

    if new_config_kwargs.get("routed_scaling_factor") is None:
        new_config_kwargs["routed_scaling_factor"] = 1.0

    rope_theta = new_config_kwargs.pop("rope_theta", 5_000_000.0)
    swa_rope_theta = new_config_kwargs.pop("swa_rope_theta", 10_000.0)
    partial_rotary_factor = new_config_kwargs.pop("partial_rotary_factor", 0.334)
    rope_scaling = new_config_kwargs.pop("rope_scaling", None)
    rope_parameters = {
        "full_attention": {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": partial_rotary_factor,
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": swa_rope_theta,
            "partial_rotary_factor": partial_rotary_factor,
        },
    }
    if rope_scaling is not None:
        rope_parameters["full_attention"].update(rope_scaling)
    new_config_kwargs["rope_parameters"] = rope_parameters

    new_config = MiMoV2FlashConfig(**new_config_kwargs)
    return new_config


def convert_mimo_v2_flash_model(input_dir, output_dir):
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
        help=("Location to write the converted `config.json`."),
    )
    args = parser.parse_args()
    convert_mimo_v2_flash_model(args.input_dir, args.output_dir)
