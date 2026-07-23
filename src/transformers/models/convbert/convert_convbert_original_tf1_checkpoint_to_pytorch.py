
import argparse
import os
from operator import attrgetter

import torch

from transformers import ConvBertConfig, ConvBertModel
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_tf_weights_in_convbert(model, config, tf_checkpoint_path):
    pass


def convert_orig_tf1_checkpoint_to_pytorch(tf_checkpoint_path, convbert_config_file, pytorch_dump_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--convbert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained ConvBERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_orig_tf1_checkpoint_to_pytorch(args.tf_checkpoint_path, args.convbert_config_file, args.pytorch_dump_path)
