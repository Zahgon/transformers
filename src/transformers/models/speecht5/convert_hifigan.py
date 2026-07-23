
import argparse

import numpy as np
import torch

from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig, logging


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.speecht5")


def load_weights(checkpoint, hf_model, config):
    pass


@torch.no_grad()
def convert_hifigan_checkpoint(
    checkpoint_path,
    stats_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--stats_path", required=True, default=None, type=str, help="Path to stats.npy file")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the Hugging Face hub."
    )

    args = parser.parse_args()
    convert_hifigan_checkpoint(
        args.checkpoint_path,
        args.stats_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.push_to_hub,
    )
