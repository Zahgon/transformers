import argparse
import json
import os

import tensorstore as ts
import torch
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict
from tensorflow.io import gfile

from transformers.models.switch_transformers.convert_switch_transformers_original_flax_checkpoint_to_pytorch import (
    rename_keys,
)
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import convert_file_size_to_int


def rename_base_flax_keys(flax_key_tuple, flax_tensor):
    pass


def get_key_and_tensorstore_dict(layer, checkpoint_info, switch_checkpoint_path):
    pass


def rename_and_save_block(current_block, save_path):
    pass


def shard_on_the_fly(switch_checkpoint_path, dump_path, max_shard_size, dtype, weights_name: str = WEIGHTS_NAME):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--switch_t5x_checkpoint_path",
        default="/mnt/disks/disk_switch/original_checkpoints/switch-xxl-128/checkpoint_634600",
        type=str,
        required=False,
        help="Path to a directory containing a folder per layer. Follows the original Google format.",
    )
    parser.add_argument("--max_shard_size", default="10GB", required=False, help="Max shard size")
    parser.add_argument("--dtype", default="bfloat16", type=str, required=False, help="dtype of the saved model")
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/mnt/disks/disk_switch/original_checkpoints/switch-xxl-128-converted",
        type=str,
        required=False,
        help="Path to the output pytorch model.",
    )
    args = parser.parse_args()
    shard_on_the_fly(
        args.switch_t5x_checkpoint_path,
        args.pytorch_dump_folder_path,
        args.max_shard_size,
        args.dtype,
    )


def sanity_check():
    pass
