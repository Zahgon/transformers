
import argparse
import re

import h5py
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from transformers.models.moonshine.modeling_moonshine import MoonshineConfig, MoonshineForConditionalGeneration


def _get_weights(model_name):
    pass


def _read_h5_weights(group, current_key="", weights=None):
    pass


def _convert_layer_names(name, gated_mlp=False):
    pass


def _convert_weights(weights, encoder=True):
    pass


def convert_usefulsensors_moonshine_to_hf(model_name, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Path to the downloaded checkpoints")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()

    convert_usefulsensors_moonshine_to_hf(args.model_name, args.pytorch_dump_folder_path)
