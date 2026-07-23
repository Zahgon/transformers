
import argparse
import os

import torch

from transformers import ImageGPTConfig, ImageGPTForCausalLM
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_tf_weights_in_imagegpt(model, config, imagegpt_checkpoint_path):
    pass


def convert_imagegpt_checkpoint_to_pytorch(imagegpt_checkpoint_path, model_size, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagegpt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint path.",
    )
    parser.add_argument(
        "--model_size",
        default=None,
        type=str,
        required=True,
        help="Size of the model (can be either 'small', 'medium' or 'large').",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_imagegpt_checkpoint_to_pytorch(
        args.imagegpt_checkpoint_path, args.model_size, args.pytorch_dump_folder_path
    )
