
import argparse
import json
import os
import pickle
import re
from pathlib import Path

import torch
from safetensors.torch import save_file


class UnpicklerWrapper(pickle.Unpickler):
    def find_class(self, mod_name, name):
        pass


pickle.Unpickler = UnpicklerWrapper


def dict_access_multi(a_dict, keys):
    pass


def _build_neox_to_llama_perm(rotary_dim: int) -> torch.Tensor:
    pass


def _apply_rope_permute(q_or_k: torch.Tensor, blocks: int, head_dim: int, rotary_dim: int, neox_to_llama: bool = True):
    pass


def merge_qkv(
    sd_list,
    original_tp,
    num_attention_heads,
    multi_query_group_num,
    attention_dim,
    interleaved_qkv,
    convert_neox_to_llama: bool = True,
):
    pass


def merge_qkv_vit(sd_list, original_tp, num_attention_heads, multi_query_group_num, attention_dim):
    pass


def merge_glu(sd_list):
    pass


def merge_glu_vit(sd_list, original_tp=None):
    pass


def split_glu(sd, cnt, idx):
    pass


def merge_tensors(
    tp_sd,
    keys,
    original_tp,
    target_tp,
    current_tp,
    slice_dim=None,
    merge_fn=None,
):
    pass


def save_sharded_model(state_dict, output_path, max_shard_size_gb=5, num_layers=40, vision_num_layers=24):
    os.makedirs(output_path, exist_ok=True)

    layered_dict = {}
    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        layered_dict[layer_key] = {}

        for key, value in state_dict.items():
            if f"model.language_model.layers.{layer_idx}." in key:
                if isinstance(value, list):
                    assert len(value) == 1, f"{key} {value}"
                    value = value[0]
                layered_dict[layer_key][key] = value

    for layer_idx in range(vision_num_layers):
        layer_key = f"visual_layer_{layer_idx}"
        layered_dict[layer_key] = {}

        for key, value in state_dict.items():
            if f"model.visual.blocks.{layer_idx}." in key:
                layered_dict[layer_key][key] = value

    layered_dict["others"] = {}
    for key, value in state_dict.items():
        if not any(f"model.language_model.layers.{i}." in key for i in range(num_layers)) and not any(
            f"model.visual.blocks.{i}." in key for i in range(vision_num_layers)
        ):
            layered_dict["others"][key] = value

    layer_order = []
    for i in range(num_layers):
        layer_order.append(f"layer_{i}")
    for i in range(vision_num_layers):
        layer_order.append(f"visual_layer_{i}")
    layer_order.append("others")

    param_sizes = {}
    shards = []
    current_shard = {}
    current_shard_size = 0
    max_shard_size_bytes = max_shard_size_gb * 1024 * 1024 * 1024

    for layer_key in layer_order:
        layer_weights = layered_dict[layer_key]
        layer_size = sum(param.numel() * param.element_size() for param in layer_weights.values())
        if current_shard_size + layer_size > max_shard_size_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0
        for param_name, param in layer_weights.items():
            current_shard[param_name] = param
            current_shard_size += param.numel() * param.element_size()
            param_sizes[param_name] = param.numel() * param.element_size()
    if current_shard:
        shards.append(current_shard)
    index_dict = {"metadata": {"total_size": sum(param_sizes.values())}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_filename = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        shard_path = os.path.join(output_path, shard_filename)

        for param_name in shard:
            index_dict["weight_map"][param_name] = shard_filename

        save_file(shard, shard_path, metadata={"format": "pt"})
        print(f"Saved shard {i + 1}/{len(shards)}: {shard_filename}")
        print(f"  Shard size: {sum(p.numel() * p.element_size() for p in shard.values()) / (1024**3):.2f} GB")
        print(f"  Keys in shard: {len(shard)}")

    index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index_dict, f, indent=2)

    return len(shards)


def merge_tp_weights(model_path, output_path, vllm_config_path=None):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Megatron model to HuggingFace format")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Megatron model directory",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output path for HuggingFace model directory")
    parser.add_argument(
        "--config_path", type=str, help="Path to vLLM configuration file for creating HuggingFace config"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_tp_weights(args.model_path, args.output_path, args.config_path)
