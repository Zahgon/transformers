
import argparse
from pathlib import Path

import fairseq
import torch
from fairseq.models.xmod import XMODModel as FairseqXmodModel
from packaging import version

from transformers import XmodConfig, XmodForMaskedLM, XmodForSequenceClassification
from transformers.utils import logging


if version.parse(fairseq.__version__) < version.parse("0.12.2"):
    raise Exception("requires fairseq >= 0.12.2")
if version.parse(fairseq.__version__) > version.parse("2"):
    raise Exception("requires fairseq < v2")

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = "Hello, World!"
SAMPLE_LANGUAGE = "en_XX"


def convert_xmod_checkpoint_to_pytorch(
    xmod_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xmod_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    args = parser.parse_args()
    convert_xmod_checkpoint_to_pytorch(
        args.xmod_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
