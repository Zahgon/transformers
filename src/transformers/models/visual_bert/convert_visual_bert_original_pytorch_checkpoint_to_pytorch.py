
import argparse
from collections import OrderedDict
from pathlib import Path

import torch

from transformers import (
    VisualBertConfig,
    VisualBertForMultipleChoice,
    VisualBertForPreTraining,
    VisualBertForQuestionAnswering,
    VisualBertForVisualReasoning,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

rename_keys_prefix = [
    ("bert.bert", "visual_bert"),
    ("bert.cls", "cls"),
    ("bert.classifier", "cls"),
    ("token_type_embeddings_visual", "visual_token_type_embeddings"),
    ("position_embeddings_visual", "visual_position_embeddings"),
    ("projection", "visual_projection"),
]

ACCEPTABLE_CHECKPOINTS = [
    "nlvr2_coco_pre_trained.th",
    "nlvr2_fine_tuned.th",
    "nlvr2_pre_trained.th",
    "vcr_coco_pre_train.th",
    "vcr_fine_tune.th",
    "vcr_pre_train.th",
    "vqa_coco_pre_trained.th",
    "vqa_fine_tuned.th",
    "vqa_pre_trained.th",
]


def load_state_dict(checkpoint_path):
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return sd


def get_new_dict(d, config, rename_keys_prefix=rename_keys_prefix):
    pass


@torch.no_grad()
def convert_visual_bert_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("orig_checkpoint_path", type=str, help="A path to .th on local filesystem.")
    parser.add_argument("pytorch_dump_folder_path", type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_visual_bert_checkpoint(args.orig_checkpoint_path, args.pytorch_dump_folder_path)
