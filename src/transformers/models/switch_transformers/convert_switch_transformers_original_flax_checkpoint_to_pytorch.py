

import argparse
import re

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from t5x import checkpoints

from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_flax_weights_in_pytorch_model(pt_model, flax_state):
    pass


MOE_LAYER_NAME_MAPPING = {
    "/attention/": "/0/SelfAttention/",
    "/self_attention/": "/0/SelfAttention/",
    "/encoder_decoder_attention/": "/1/EncDecAttention/",
    "value": "v",
    "query": "q",
    "key": "k",
    "out": "o",
    "pre_self_attention_layer_norm": "0/layer_norm",
    "pre_cross_attention_layer_norm": "1/layer_norm",
    "pre_attention_layer_norm": "0/layer_norm",  # previously 1, but seems wrong
    "token_embedder": "shared",
    "encoder_norm": "final_layer_norm",
    "decoder_norm": "final_layer_norm",
    "relpos_bias/rel_embedding": "block/0/layer/0/SelfAttention/relative_attention_bias/weight",
    "router/router_weights/w/": "router/classifier/",
    "roer/roer_weights/w/": "router/classifier/",
    "logits_dense": "lm_head",
}


def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        layer_to_block_of_layer = r".*/layers_(\d+)"
        new_key = key
        if re.match(layer_to_block_of_layer, key):
            new_key = re.sub(r"layers_(\d+)", r"block/\1/layer", new_key)

        layer_to_block_of_layer = r"(encoder|decoder)\/"

        if re.match(layer_to_block_of_layer, key):
            groups = re.match(layer_to_block_of_layer, new_key).groups()
            if groups[0] == "encoder":
                new_key = re.sub(r"/mlp/", r"/1/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/1/layer_norm/", new_key)

            elif groups[0] == "decoder":
                new_key = re.sub(r"/mlp/", r"/2/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/2/layer_norm/", new_key)

        for old_key, temp_key in MOE_LAYER_NAME_MAPPING.items():
            if old_key in new_key:
                new_key = new_key.replace(old_key, temp_key)

        print(f"{key} -> {new_key}")
        s_dict[new_key] = s_dict.pop(key)

    if "encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight" in s_dict:
        s_dict["encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"] = s_dict[
            "encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"
        ].T
    if "decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight" in s_dict:
        s_dict["decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"] = s_dict[
            "decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"
        ].T

    for key in list(s_dict.keys()):
        if "expert" in key:
            num_experts = s_dict[key].shape[0]
            expert_weihts = s_dict[key]
            for idx in range(num_experts):
                s_dict[key.replace("expert/", f"experts/expert_{idx}/")] = expert_weihts[idx]
                print(f"{key} -> {key.replace('expert/', f'experts/expert_{idx}/')}")

            s_dict.pop(key)

    return s_dict


GIN_TO_CONFIG_MAPPING = {
    "NUM_ENCODER_LAYERS": "num_layers",
    "NUM_DECODER_LAYERS": "num_decoder_layers",
    "NUM_HEADS": "num_heads",
    "HEAD_DIM": "d_kv",
    "EMBED_DIM": "d_model",
    "MLP_DIM": "d_ff",
    "NUM_SELECTED_EXPERTS": "num_selected_experts",
    "NUM_ENCODER_SPARSE_LAYERS": "num_sparse_encoder_layers",
    "NUM_DECODER_SPARSE_LAYERS": "num_sparse_decoder_layers",
    "dense.MlpBlock.activations": "feed_forward_proj",
}


def convert_gin_to_config(gin_file, num_experts):
    pass


def convert_flax_checkpoint_to_pytorch(
    flax_checkpoint_path, config_file, gin_file=None, pytorch_dump_path="./", num_experts=8
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--switch_t5x_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained SwitchTransformers model. \nThis specifies the"
            " model architecture. If not provided, a `gin_file` has to be provided."
        ),
    )
    parser.add_argument(
        "--gin_file",
        default=None,
        type=str,
        required=False,
        help="Path to the gin config file. If not provided, a `config_file` has to be passed   ",
    )
    parser.add_argument(
        "--config_name", default=None, type=str, required=False, help="Config name of SwitchTransformers model."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output pytorch model."
    )
    parser.add_argument("--num_experts", default=8, type=int, required=False, help="Number of experts")
    args = parser.parse_args()
    convert_flax_checkpoint_to_pytorch(
        args.switch_t5x_checkpoint_path,
        args.config_name,
        args.gin_file,
        args.pytorch_dump_folder_path,
        args.num_experts,
    )
