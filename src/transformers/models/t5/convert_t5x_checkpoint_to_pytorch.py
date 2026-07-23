
import argparse
import collections

import torch
from flax import traverse_util
from t5x import checkpoints

from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration
from transformers.utils import logging


logging.set_verbosity_info()


def t5x_attention_lookup(params, i, prefix, layer_name="attention"):
    pass


def t5x_mlp_lookup(params, i, prefix, split_mlp_wi=False):
    pass


def t5x_layer_norm_lookup(params, i, prefix, layer_name):
    pass


def convert_t5x_to_pytorch(variables: dict, *, num_layers: int, num_decoder_layers: int, is_encoder_only: bool):
    pass


def make_state_dict(converted_params, is_encoder_only: bool):
    pass


def load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only):
    pass


def convert_t5x_checkpoint_to_pytorch(
    t5x_checkpoint_path, config_file, pytorch_dump_path, is_encoder_only: bool = False
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a native T5X checkpoint into a PyTorch checkpoint.")
    parser.add_argument(
        "--t5x_checkpoint_path", default=None, type=str, required=True, help="Path to the T5X checkpoint."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained T5 model.\nThis specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--is_encoder_only", action="store_true", help="Check if the model is encoder-decoder model", default=False
    )
    args = parser.parse_args()
    convert_t5x_checkpoint_to_pytorch(
        args.t5x_checkpoint_path, args.config_file, args.pytorch_dump_path, args.is_encoder_only
    )
