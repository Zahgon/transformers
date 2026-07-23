
import argparse
import json
from collections import OrderedDict
from functools import partial
from pathlib import Path

import timm
import torch
from huggingface_hub import hf_hub_download

from transformers import LevitConfig, LevitForImageClassificationWithTeacher, LevitImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger()


def convert_weight_and_push(
    hidden_sizes: int, name: str, config: LevitConfig, save_directory: Path, push_to_hub: bool = True
):
    pass


def convert_weights_and_push(save_directory: Path, model_name: str | None = None, push_to_hub: bool = True):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="The name of the model you wish to convert, it must be one of the supported Levit* architecture,",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="levit-dump-folder/",
        type=Path,
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")
    parser.add_argument(
        "--no-push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Do not push model and image processor to the hub",
    )

    args = parser.parse_args()
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)
