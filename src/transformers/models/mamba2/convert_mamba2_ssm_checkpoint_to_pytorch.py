
import argparse
import json
from functools import partial
from os import path

import torch
from safetensors import safe_open
from safetensors.torch import save_model

from transformers import GPTNeoXTokenizerFast, LlamaTokenizerFast, Mamba2Config, Mamba2ForCausalLM


def load_state_dict_from_safetensors(mamba2_checkpoint_path: str, ckpt_name: str) -> dict[str, torch.Tensor]:
    pass


def load_state_dict_from_torch(mamba2_checkpoint_path: str, ckpt_name: str) -> dict[str, torch.Tensor]:
    pass


def convert_ssm_config_to_hf_config(config_ssm: dict, mamba2_model_dict: dict) -> Mamba2Config:
    pass


def load_and_save_tokenizer(
    mamba2_model_type: str,
    output_dir: str,
    tokenizer_model_path: str | None = None,
) -> None:
    pass


_MAMBA2_MODELS_DICT = {
    "codestral": {
        "hidden_size": "dim",
        "num_hidden_layers": "n_layers",
        "n_groups": "n_groups",
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "config_name": "params.json",
        "load_state_dict": partial(load_state_dict_from_safetensors, ckpt_name="consolidated.safetensors"),
        "load_and_save_tokenizer": partial(load_and_save_tokenizer, "codestral"),
    },
    "mamba_ssm": {
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layer",
        "n_groups": "ngroups",
        "bos_token_id": 0,
        "pad_token_id": 0,
        "eos_token_id": 0,
        "config_name": "config.json",
        "load_state_dict": partial(load_state_dict_from_torch, ckpt_name="pytorch_model.bin"),
        "load_and_save_tokenizer": partial(load_and_save_tokenizer, "mamba_ssm"),
    },
}


def convert_mamba2_checkpoint_file_to_huggingface_model_file(
    mamba2_checkpoint_path: str,
    mamba2_model_type: str,
    precision: str,
    output_dir: str,
    tokenizer_model_path: str | None = None,
) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--mamba2_checkpoint_directory",
        type=str,
        required=True,
        help="Path to a directory containing the `pytorch_model.bin` or `.safetensors` mamba2_ssm checkpoint file to be converted.",
    )
    parser.add_argument(
        "-m",
        "--mamba2_model_type",
        type=str,
        default="mamba_ssm",
        const="mamba_ssm",
        required=True,
        choices=("codestral", "mamba_ssm"),
        help="The model type the conversion will be performed on. Can choose from either `codestral` or `mamba_ssm`.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="fp16",
        const="fp16",
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
        help="Path to a `codestral` tokenizer file.",
    )
    args = parser.parse_args()

    convert_mamba2_checkpoint_file_to_huggingface_model_file(
        args.mamba2_checkpoint_directory,
        args.mamba2_model_type,
        args.precision,
        args.output_dir,
        args.tokenizer_model_path,
    )
