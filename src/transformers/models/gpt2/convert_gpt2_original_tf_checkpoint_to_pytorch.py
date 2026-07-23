
import argparse
import os

import torch

from transformers import GPT2Config, GPT2Model
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    pass


def convert_gpt2_checkpoint_to_pytorch(gpt2_checkpoint_path, gpt2_config_file, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpt2_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--gpt2_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained OpenAI model. \n"
            "This specifies the model architecture."
        ),
    )
    args = parser.parse_args()
    convert_gpt2_checkpoint_to_pytorch(args.gpt2_checkpoint_path, args.gpt2_config_file, args.pytorch_dump_folder_path)
