
import argparse
import collections

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    Gemma2Config,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    SiglipImageProcessor,
)
from transformers.tokenization_utils_base import AddedToken
from transformers.utils import logging


device = "cpu"

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


PALIGEMMA2_VARIANTS = ["2b-224", "2b-448", "2b-896", "9b-224", "9b-448", "9b-896", "27b-224", "27b-448", "27b-896"]
VARIANT_CONFIGS = {
    "2b": {
        "num_positions": 256,
        "hidden_size": 2304,
        "num_hidden_layers": 26,
        "intermediate_size": 9216,
        "num_key_value_heads": 4,
        "num_attention_heads": 8,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
    },
    "9b": {
        "num_positions": 1024,
        "hidden_size": 3584,
        "num_hidden_layers": 42,
        "intermediate_size": 14336,
        "num_key_value_heads": 8,
        "num_attention_heads": 16,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
    },
    "27b": {
        "num_positions": 4096,
        "hidden_size": 4608,
        "num_hidden_layers": 46,
        "intermediate_size": 36864,
        "num_key_value_heads": 16,
        "num_attention_heads": 32,
        "head_dim": 128,
        "query_pre_attn_scalar": 4608 // 32,  # scaling is different for the 28b
    },
}

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def get_paligemma2_config(variant: str, precision: str):
    pass


def slice_state_dict(state_dict, config):
    pass


def flatten_nested_dict(params, parent_key="", sep="/", precision: int = "float32"):
    pass


@torch.no_grad()
def convert_paligemma2_checkpoint(
    checkpoint_path,
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
        default="2b-224",
        choices=PALIGEMMA2_VARIANTS,
        type=str,
        help="String identifier of the paligemma2 variant to convert.",
    )

    parser.add_argument(
        "--do_convert_weights", action="store_true", help="Whether or not to reload and convert the weights."
    )

    args = parser.parse_args()
    convert_paligemma2_checkpoint(
        checkpoint_path=args.checkpoint_path,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        variant=args.variant,
        precision=args.precision,
        do_convert_weights=args.do_convert_weights,
    )
