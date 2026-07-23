
import argparse
import json
import os

import torch

from transformers import OpenAIGPTConfig, OpenAIGPTModel
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_tf_weights_in_openai_gpt(model, config, openai_checkpoint_folder_path):
    pass


def convert_openai_checkpoint_to_pytorch(openai_checkpoint_folder_path, openai_config_file, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openai_checkpoint_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint path.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--openai_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained OpenAI model. \n"
            "This specifies the model architecture."
        ),
    )
    args = parser.parse_args()
    convert_openai_checkpoint_to_pytorch(
        args.openai_checkpoint_folder_path, args.openai_config_file, args.pytorch_dump_folder_path
    )
