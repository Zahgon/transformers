
import argparse
import os

import torch

from transformers import (
    TapasConfig,
    TapasForMaskedLM,
    TapasForQuestionAnswering,
    TapasForSequenceClassification,
    TapasModel,
    TapasTokenizer,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_tf_weights_in_tapas(model, config, tf_checkpoint_path):
    pass


def convert_tf_checkpoint_to_pytorch(
    task, reset_position_index_per_cell, tf_checkpoint_path, tapas_config_file, pytorch_dump_path
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", default="SQA", type=str, help="Model task for which to convert a checkpoint. Defaults to SQA."
    )
    parser.add_argument(
        "--reset_position_index_per_cell",
        default=False,
        action="store_true",
        help="Whether to use relative position embeddings or not. Defaults to True.",
    )
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--tapas_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained TAPAS model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.task,
        args.reset_position_index_per_cell,
        args.tf_checkpoint_path,
        args.tapas_config_file,
        args.pytorch_dump_path,
    )
