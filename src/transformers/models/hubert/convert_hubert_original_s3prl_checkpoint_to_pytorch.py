
import argparse

import torch

from transformers import HubertConfig, HubertForSequenceClassification, Wav2Vec2FeatureExtractor, logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SUPPORTED_MODELS = ["UtteranceLevel"]


@torch.no_grad()
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", default=None, type=str, help="Name of the huggingface pretrained base model."
    )
    parser.add_argument("--config_path", default=None, type=str, help="Path to the huggingface classifier config.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the s3prl checkpoint.")
    parser.add_argument("--model_dump_path", default=None, type=str, help="Path to the final converted model.")
    args = parser.parse_args()
    convert_s3prl_checkpoint(args.base_model_name, args.config_path, args.checkpoint_path, args.model_dump_path)
