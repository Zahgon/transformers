
import argparse

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms as T

from transformers import (
    LayoutLMv3ImageProcessor,
    UdopConfig,
    UdopForConditionalGeneration,
    UdopProcessor,
    UdopTokenizer,
)
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def original_transform(image, image_size=224):
    pass


def get_image():
    pass


def prepare_dummy_inputs(tokenizer, image_processor):
    pass


def convert_udop_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="udop-large",
        type=str,
        choices=["udop-large", "udop-large-512", "udop-large-512-300k"],
        help=("Name of the UDOP model you'd like to convert."),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_udop_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
