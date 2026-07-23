

import argparse

import tensorflow as tf
import torch

from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertPooler,
    BertSelfAttention,
    BertSelfOutput,
)
from transformers.utils import logging


logging.set_verbosity_info()


def convert_checkpoint_to_pytorch(tf_checkpoint_path: str, config_path: str, pytorch_dump_path: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path", type=str, required=True, help="Path to the TensorFlow Token Dropping checkpoint path."
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
        help="Path to the output PyTorch model.",
    )
    args = parser.parse_args()
    convert_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
