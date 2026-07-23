
import argparse
import os
import pathlib

import fairseq
import torch
from fairseq.modules import TransformerSentenceEncoderLayer
from packaging import version

from transformers import (
    Data2VecTextConfig,
    Data2VecTextForMaskedLM,
    Data2VecTextForSequenceClassification,
    Data2VecTextModel,
)
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)

from transformers.utils import logging


if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = "Hello world! cécé herlolip"


def convert_data2vec_checkpoint_to_pytorch(
    data2vec_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    args = parser.parse_args()
    convert_data2vec_checkpoint_to_pytorch(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
