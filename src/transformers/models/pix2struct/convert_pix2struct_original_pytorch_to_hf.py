import argparse
import os
import re

import torch
from flax.traverse_util import flatten_dict
from t5x import checkpoints

from transformers import (
    AutoTokenizer,
    Pix2StructConfig,
    Pix2StructForConditionalGeneration,
    Pix2StructImageProcessor,
    Pix2StructProcessor,
    Pix2StructTextConfig,
    Pix2StructVisionConfig,
)


def get_flax_param(t5x_checkpoint_path):
    pass


def rename_and_convert_flax_params(flax_dict):
    pass


def convert_pix2struct_original_pytorch_checkpoint_to_hf(
    t5x_checkpoint_path, pytorch_dump_folder_path, use_large=False, is_vqa=False
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5x_checkpoint_path", default=None, type=str, help="Path to the original T5x checkpoint.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--use_large", action="store_true", help="Use large model.")
    parser.add_argument("--is_vqa", action="store_true", help="Use large model.")
    args = parser.parse_args()

    convert_pix2struct_original_pytorch_checkpoint_to_hf(
        args.t5x_checkpoint_path, args.pytorch_dump_folder_path, args.use_large
    )
