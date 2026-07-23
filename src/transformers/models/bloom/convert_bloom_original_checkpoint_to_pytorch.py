
import argparse
import json
import os
import re

import torch

from transformers import BloomConfig, BloomModel
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging


logging.set_verbosity_info()

WEIGHTS_TO_AVERAGE_ENDSWITH = [
    "word_embeddings_layernorm.weight",
    "word_embeddings_layernorm.bias",
    "input_layernorm.weight",
    "input_layernorm.bias",
    "post_attention_layernorm.weight",
    "post_attention_layernorm.bias",
    "self_attention.dense.bias",
    "mlp.dense_4h_to_h.bias",
    "ln_f.weight",
    "ln_f.bias",
]

WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN = [
    "mlp.dense_4h_to_h.weight",
    "self_attention.dense.weight",
]


def layer_name_mapping(key, file):
    pass


def get_dtype_size(dtype):
    pass


def convert_bloom_checkpoint_to_pytorch(
    bloom_checkpoint_path, bloom_config_file, pytorch_dump_folder_path, shard_model, pretraining_tp
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bloom_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the Megatron-LM checkpoint path.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--bloom_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--shard_model",
        action="store_true",
        help="An optional setting to shard the output model \nThis enables sharding the converted checkpoint",
    )
    parser.add_argument(
        "--pretraining_tp",
        default=4,
        type=int,
        help="Pretraining TP rank that has been used when training the model in Megatron-LM \n",
    )
    args = parser.parse_args()
    convert_bloom_checkpoint_to_pytorch(
        args.bloom_checkpoint_path,
        args.bloom_config_file,
        args.pytorch_dump_folder_path,
        args.shard_model,
        args.pretraining_tp,
    )
