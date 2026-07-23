
import argparse
import collections

import torch
from numpy import load

from transformers import (
    AutoTokenizer,
    GemmaTokenizer,
    GemmaTokenizerFast,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    SiglipImageProcessor,
)
from transformers.tokenization_utils_base import AddedToken
from transformers.utils import logging


device = "cuda"  # "cpu"

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


PALIGEMMA_VARIANTS = ["2b-test", "3b-224px", "3b-448px", "3b-896px"]


def get_paligemma_config(variant: str, precision: str):
    pass


def slice_state_dict(state_dict, config):
    pass


def flatten_nested_dict(params, parent_key="", sep="/"):
    pass


@torch.no_grad()
def convert_paligemma_checkpoint(
    checkpoint_path,
    tokenizer_model_file,
    pytorch_dump_folder_path,
    variant: str,
    precision: str,
    do_convert_weights=False,
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to the .npz checkpoint",
    )

    parser.add_argument(
        "--tokenizer_model_file",
        required=True,
        type=str,
        help="Path to the sentencepiece tokenizer.model file",
    )

    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=str,
        help="Path to the output directory where model and processor will be saved.",
    )

    parser.add_argument(
        "--precision",
        choices=["float32", "bfloat16", "float16"],
        type=str,
        help="Precision identifier for model conversion - should match the base checkpoint precision.",
    )

    parser.add_argument(
        "--variant",
        default="2b-test",
        choices=PALIGEMMA_VARIANTS,
        type=str,
        help="String identifier of the paligemma variant to convert.",
    )

    parser.add_argument(
        "--do_convert_weights", action="store_true", help="Whether or not to reload and convert the weights."
    )

    args = parser.parse_args()
    convert_paligemma_checkpoint(
        checkpoint_path=args.checkpoint_path,
        tokenizer_model_file=args.tokenizer_model_file,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        variant=args.variant,
        precision=args.precision,
        do_convert_weights=args.do_convert_weights,
    )
