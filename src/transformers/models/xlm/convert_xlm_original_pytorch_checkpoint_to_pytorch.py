
import argparse
import json

import numpy
import torch

from transformers.models.xlm.tokenization_xlm import VOCAB_FILES_NAMES
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging


logging.set_verbosity_info()


def convert_xlm_checkpoint_to_pytorch(xlm_checkpoint_path, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlm_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_xlm_checkpoint_to_pytorch(args.xlm_checkpoint_path, args.pytorch_dump_folder_path)
