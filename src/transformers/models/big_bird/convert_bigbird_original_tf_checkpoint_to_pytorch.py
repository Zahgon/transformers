
import argparse
import math
import os

import torch

from transformers import BigBirdConfig, BigBirdForPreTraining, BigBirdForQuestionAnswering
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


_TRIVIA_QA_MAPPING = {
    "big_bird_attention": "attention/self",
    "output_layer_norm": "output/LayerNorm",
    "attention_output": "attention/output/dense",
    "output": "output/dense",
    "self_attention_layer_norm": "attention/output/LayerNorm",
    "intermediate": "intermediate/dense",
    "word_embeddings": "bert/embeddings/word_embeddings",
    "position_embedding": "bert/embeddings/position_embeddings",
    "type_embeddings": "bert/embeddings/token_type_embeddings",
    "embeddings": "bert/embeddings",
    "layer_normalization": "output/LayerNorm",
    "layer_norm": "LayerNorm",
    "trivia_qa_head": "qa_classifier",
    "dense": "intermediate/dense",
    "dense_1": "qa_outputs",
}


def load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=False):
    pass


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, big_bird_config_file, pytorch_dump_path, is_trivia_qa):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--big_bird_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--is_trivia_qa", action="store_true", help="Whether to convert a model with a trivia_qa head."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.big_bird_config_file, args.pytorch_dump_path, args.is_trivia_qa
    )
