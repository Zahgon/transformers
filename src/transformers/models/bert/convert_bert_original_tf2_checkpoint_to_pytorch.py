

import argparse
import os
import re

import tensorflow as tf
import torch

from transformers import BertConfig, BertModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_tf2_weights_in_bert(model, tf_checkpoint_path, config):
    pass


def convert_tf2_checkpoint_to_pytorch(tf_checkpoint_path, config_path, pytorch_dump_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path", type=str, required=True, help="Path to the TensorFlow 2.x checkpoint path."
    )
    parser.add_argument(
        "--bert_config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the BERT model. This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model (must include filename).",
    )
    args = parser.parse_args()
    convert_tf2_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
