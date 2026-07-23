
import argparse

import bros  # original repo
import torch

from transformers import BrosConfig, BrosModel, BrosProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_configs(model_name):
    pass


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "embeddings.bbox_sinusoid_emb.inv_freq",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    if name == "embeddings.bbox_projection.weight":
        name = "bbox_embeddings.bbox_projection.weight"

    if name == "embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq":
        name = "bbox_embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq"

    if name == "embeddings.bbox_sinusoid_emb.y_pos_emb.inv_freq":
        name = "bbox_embeddings.bbox_sinusoid_emb.y_pos_emb.inv_freq"

    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)
        orig_state_dict[rename_key(key)] = val

    remove_ignore_keys_(orig_state_dict)

    return orig_state_dict


def convert_bros_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="jinho8345/bros-base-uncased",
        required=False,
        type=str,
        help="Name of the original model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_bros_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
