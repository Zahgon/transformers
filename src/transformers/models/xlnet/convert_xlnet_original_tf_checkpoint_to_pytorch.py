
import argparse
import os

import torch

from transformers import (
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetForSequenceClassification,
    XLNetLMHeadModel,
)
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging


GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
    pass


def load_tf_weights_in_xlnet(model, config, tf_path):
    pass


def convert_xlnet_checkpoint_to_pytorch(
    tf_checkpoint_path, bert_config_file, pytorch_dump_folder_path, finetuning_task=None
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--xlnet_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained XLNet model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to store the PyTorch model or dataset/vocab.",
    )
    parser.add_argument(
        "--finetuning_task",
        default=None,
        type=str,
        help="Name of a task on which the XLNet TensorFlow model was fine-tuned",
    )
    args = parser.parse_args()
    print(args)

    convert_xlnet_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.xlnet_config_file, args.pytorch_dump_folder_path, args.finetuning_task
    )
