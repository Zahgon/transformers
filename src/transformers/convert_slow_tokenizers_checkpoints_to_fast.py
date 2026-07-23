
import argparse
import os
from pathlib import Path

import transformers

from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
from .utils import logging


logging.set_verbosity_info()

logger = logging.get_logger(__name__)


TOKENIZER_CLASSES = {}
for name in SLOW_TO_FAST_CONVERTERS:
    if name == "Phi3Tokenizer":
        tokenizer_class_name = "LlamaTokenizerFast"
    elif name == "ElectraTokenizer":
        tokenizer_class_name = "BertTokenizerFast"
    else:
        tokenizer_class_name = name + "Fast"

    try:
        TOKENIZER_CLASSES[name] = getattr(transformers, tokenizer_class_name)
    except AttributeError:
        pass


def convert_slow_checkpoint_to_fast(tokenizer_name, checkpoint_name, dump_path, force_download):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Path to output generated fast tokenizer files."
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help=(
            f"Optional tokenizer type selected in the list of {list(TOKENIZER_CLASSES.keys())}. If not given, will "
            "download and convert all the checkpoints from AWS."
        ),
    )
    parser.add_argument(
        "--checkpoint_name",
        default=None,
        type=str,
        help="Optional checkpoint name. If not given, will download and convert the canonical checkpoints from AWS.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download checkpoints.",
    )
    args = parser.parse_args()

    convert_slow_checkpoint_to_fast(args.tokenizer_name, args.checkpoint_name, args.dump_path, args.force_download)
