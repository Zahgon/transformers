

import argparse
import json
import os
import re
from collections import OrderedDict
from os.path import basename, dirname

import fairseq
import torch
from fairseq import hub_utils
from fairseq.data.dictionary import Dictionary

from transformers import FSMTConfig, FSMTForConditionalGeneration
from transformers.models.fsmt.tokenization_fsmt import VOCAB_FILES_NAMES
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from transformers.utils import WEIGHTS_NAME, logging


logging.set_verbosity_warning()

json_indent = 2

best_score_hparams = {
    "wmt19-ru-en": {"length_penalty": 1.1},
    "wmt19-en-ru": {"length_penalty": 1.15},
    "wmt19-en-de": {"length_penalty": 1.0},
    "wmt19-de-en": {"length_penalty": 1.1},
    "wmt16-en-de-dist-12-1": {"length_penalty": 0.6},
    "wmt16-en-de-dist-6-1": {"length_penalty": 0.6},
    "wmt16-en-de-12-1": {"length_penalty": 0.8},
    "wmt19-de-en-6-6-base": {"length_penalty": 0.6},
    "wmt19-de-en-6-6-big": {"length_penalty": 0.6},
}

org_names = {}
for m in ["wmt19-ru-en", "wmt19-en-ru", "wmt19-en-de", "wmt19-de-en"]:
    org_names[m] = "facebook"
for m in [
    "wmt16-en-de-dist-12-1",
    "wmt16-en-de-dist-6-1",
    "wmt16-en-de-12-1",
    "wmt19-de-en-6-6-base",
    "wmt19-de-en-6-6-big",
]:
    org_names[m] = "allenai"


def rewrite_dict_keys(d):
    pass


def convert_fsmt_checkpoint_to_pytorch(fsmt_checkpoint_path, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fsmt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help=(
            "Path to the official PyTorch checkpoint file which is expected to reside in the dump dir with dicts,"
            " bpecodes, etc."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_fsmt_checkpoint_to_pytorch(args.fsmt_checkpoint_path, args.pytorch_dump_folder_path)
