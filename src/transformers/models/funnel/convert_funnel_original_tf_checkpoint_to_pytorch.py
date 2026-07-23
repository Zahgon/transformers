
import argparse
import os

import torch

from transformers import FunnelBaseModel, FunnelConfig, FunnelModel
from transformers.models.funnel.modeling_funnel import FunnelPositionwiseFFN, FunnelRelMultiheadAttention
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_tf_weights_in_funnel(model, config, tf_checkpoint_path):
    pass


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, base_model):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--base_model", action="store_true", help="Whether you want just the base model (no decoder) or not."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.base_model
    )
