
import argparse
import gc
import json
import os
import re

import torch
from huggingface_hub import hf_hub_download, split_torch_state_dict_into_shards

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, RwkvConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME


NUM_HIDDEN_LAYERS_MAPPING = {
    "169M": 12,
    "430M": 24,
    "1B5": 24,
    "3B": 32,
    "7B": 32,
    "14B": 40,
}

HIDDEN_SIZE_MAPPING = {
    "169M": 768,
    "430M": 1024,
    "1B5": 2048,
    "3B": 2560,
    "7B": 4096,
    "14B": 5120,
}


def convert_state_dict(state_dict):
    state_dict_keys = list(state_dict.keys())
    for name in state_dict_keys:
        weight = state_dict.pop(name)
        if name.startswith("emb."):
            name = name.replace("emb.", "embeddings.")
        if name.startswith("blocks.0.ln0"):
            name = name.replace("blocks.0.ln0", "blocks.0.pre_ln")
        name = re.sub(r"blocks\.(\d+)\.att", r"blocks.\1.attention", name)
        name = re.sub(r"blocks\.(\d+)\.ffn", r"blocks.\1.feed_forward", name)
        if name.endswith(".time_mix_k"):
            name = name.replace(".time_mix_k", ".time_mix_key")
        if name.endswith(".time_mix_v"):
            name = name.replace(".time_mix_v", ".time_mix_value")
        if name.endswith(".time_mix_r"):
            name = name.replace(".time_mix_r", ".time_mix_receptance")

        if name != "head.weight":
            name = "rwkv." + name

        state_dict[name] = weight
    return state_dict


def convert_rmkv_checkpoint_to_hf_format(
    repo_id, checkpoint_file, output_dir, size=None, tokenizer_file=None, push_to_hub=False, model_name=None
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id", default=None, type=str, required=True, help="Repo ID from which to pull the checkpoint."
    )
    parser.add_argument(
        "--checkpoint_file", default=None, type=str, required=True, help="Name of the checkpoint file in the repo."
    )
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True, help="Where to save the converted model."
    )
    parser.add_argument(
        "--tokenizer_file",
        default=None,
        type=str,
        help="Path to the tokenizer file to use (if not provided, only the model is converted).",
    )
    parser.add_argument(
        "--size",
        default=None,
        type=str,
        help="Size of the model. Will be inferred from the `checkpoint_file` if not passed.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push to the Hub the converted model.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Name of the pushed model on the Hub, including the username / organization.",
    )

    args = parser.parse_args()
    convert_rmkv_checkpoint_to_hf_format(
        args.repo_id,
        args.checkpoint_file,
        args.output_dir,
        size=args.size,
        tokenizer_file=args.tokenizer_file,
        push_to_hub=args.push_to_hub,
        model_name=args.model_name,
    )
