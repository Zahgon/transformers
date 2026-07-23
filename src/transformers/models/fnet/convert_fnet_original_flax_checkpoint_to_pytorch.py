
import argparse

import torch
from flax.training.checkpoints import restore_checkpoint

from transformers import FNetConfig, FNetForPreTraining
from transformers.utils import logging


logging.set_verbosity_info()


def convert_flax_checkpoint_to_pytorch(flax_checkpoint_path, fnet_config_file, save_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flax_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--fnet_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained FNet model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument("--save_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()
    convert_flax_checkpoint_to_pytorch(args.flax_checkpoint_path, args.fnet_config_file, args.save_path)
