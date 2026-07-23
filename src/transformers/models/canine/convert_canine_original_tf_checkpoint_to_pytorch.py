
import argparse
import os

import torch

from transformers import CanineConfig, CanineModel, CanineTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_tf_weights_in_canine(model, config, tf_checkpoint_path):
    pass


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, pytorch_dump_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint. Should end with model.ckpt",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to a folder where the PyTorch model will be placed.",
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.pytorch_dump_path)
