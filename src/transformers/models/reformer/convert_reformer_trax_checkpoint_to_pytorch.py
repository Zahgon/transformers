
import argparse
import os
import pickle

import numpy as np
import torch
from torch import nn

from transformers import ReformerConfig, ReformerModelWithLMHead
from transformers.utils import logging

from ...utils import strtobool


logging.set_verbosity_info()


def set_param(torch_layer, weight, bias=None):
    pass


def set_layer_weights_in_torch_lsh(weights, torch_layer, hidden_size):
    pass


def set_layer_weights_in_torch_local(weights, torch_layer, hidden_size):
    pass


def set_block_weights_in_torch(weights, torch_block, hidden_size):
    pass


def set_model_weights_in_torch(weights, torch_model, hidden_size):
    pass


def convert_trax_checkpoint_to_pytorch(trax_model_pkl_path, config_file, pytorch_dump_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trax_model_pkl_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint path.\n"
        "Given the files are in the pickle format, please be wary of passing it files you trust.",
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained Reformer model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_trax_checkpoint_to_pytorch(args.trax_model_pkl_path, args.config_file, args.pytorch_dump_path)
