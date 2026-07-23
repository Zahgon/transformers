
import argparse
import json
import os
import re
from os import path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file

from transformers import AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from .configuration_bamba import BambaConfig


def convert_state_dict_from_mamba_ssm(original_sd: dict) -> dict[str, torch.Tensor]:
    pass


def convert_ssm_config_to_hf_config(
    config_ssm: dict,
    **kwargs,
) -> BambaConfig:
    pass


def save_single_safetensor(
    state_dict: dict,
    save_directory: str,
    metadata: dict,
):
    pass


def save_sharded_safetensors(
    state_dict: dict,
    save_directory: str,
    metadata: dict,
    max_shard_size: int | str = "5GB",
):
    pass


def convert_mamba_ssm_checkpoint_file_to_huggingface_model_file(
    mamba_ssm_checkpoint_path: str,
    precision: str,
    output_dir: str,
    tokenizer_path: str | None = None,
    save_model: bool | str = True,
) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--mamba_ssm_checkpoint_directory",
        type=str,
        required=True,
        help="Path to a directory containing the `pytorch_model.bin` mamba_ssm checkpoint file to be converted.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="fp16",
        required=True,
        choices=("fp32", "fp16", "bf16"),
        help="The precision the model will be saved in. Select from fp32, fp16 or bf16.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to directory to save the converted output model to."
    )
    parser.add_argument(
        "-t",
        "--tokenizer_model_path",
        type=str,
        default=None,
        required=False,
        help="Path to a the tokenizer file.",
    )
    args = parser.parse_args()

    convert_mamba_ssm_checkpoint_file_to_huggingface_model_file(
        args.mamba_ssm_checkpoint_directory,
        args.precision,
        args.output_dir,
        save_model="sharded",
    )
