
import argparse
import json
import math

import torch

from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM
from transformers.utils import logging
from transformers.utils.import_utils import is_mamba_ssm_available


if is_mamba_ssm_available():
    from mamba_ssm.models.config_mamba import MambaConfig as MambaConfigSSM
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    def convert_ssm_config_to_hf_config(config_ssm: MambaConfigSSM) -> MambaConfig:
        pass


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def convert_mamba_ssm_checkpoint_to_huggingface_model(
    original_state_dict: dict, original_ssm_config_dict: dict
) -> tuple[MambaForCausalLM, AutoTokenizer]:
    pass


def validate_converted_model(
    original_state_dict: dict, original_ssm_config_dict: dict, hf_model: MambaForCausalLM, tokenizer: AutoTokenizer
) -> None:
    pass


def convert_mamba_checkpoint_file_to_huggingface_model_file(
    mamba_checkpoint_path: str, config_json_file: str, output_dir: str
) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--mamba_checkpoint_file",
        type=str,
        required=True,
        help="Path to a `pytorch_model.bin` mamba_ssm checkpoint file to be converted.",
    )
    parser.add_argument(
        "-c",
        "--config_json_file",
        type=str,
        required=True,
        help="Path to a `config.json` file corresponding to a MambaConfig of the original mamba_ssm model.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to directory to save the converted output model to."
    )
    args = parser.parse_args()

    convert_mamba_checkpoint_file_to_huggingface_model_file(
        args.mamba_checkpoint_file, args.config_json_file, args.output_dir
    )
