
import argparse

import torch
from huggingface_hub import hf_hub_download

from transformers import AutoTokenizer, RobertaPreLayerNormConfig, RobertaPreLayerNormForMaskedLM
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def convert_roberta_prelayernorm_checkpoint_to_pytorch(checkpoint_repo: str, pytorch_dump_folder_path: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-repo",
        default=None,
        type=str,
        required=True,
        help="Path the official PyTorch dump, e.g. 'andreasmadsen/efficient_mlm_m0.40'.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_roberta_prelayernorm_checkpoint_to_pytorch(args.checkpoint_repo, args.pytorch_dump_folder_path)
