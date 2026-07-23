
import argparse
from pathlib import Path

import torch
import yaml

from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig, logging


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")


def load_weights(checkpoint, hf_model, config):
    pass


def remap_hifigan_yaml_config(yaml_config_path):
    pass


@torch.no_grad()
def convert_hifigan_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    yaml_config_path=None,
    repo_id=None,
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--yaml_config_path", default=None, type=str, help="Path to config.yaml of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the Hugging Face hub."
    )

    args = parser.parse_args()
    convert_hifigan_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.yaml_config_path,
        args.push_to_hub,
    )
