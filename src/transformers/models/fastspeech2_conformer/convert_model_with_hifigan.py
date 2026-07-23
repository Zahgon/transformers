
import argparse

import torch

from transformers import (
    FastSpeech2ConformerConfig,
    FastSpeech2ConformerHifiGan,
    FastSpeech2ConformerHifiGanConfig,
    FastSpeech2ConformerModel,
    FastSpeech2ConformerWithHifiGan,
    FastSpeech2ConformerWithHifiGanConfig,
    logging,
)

from .convert_fastspeech2_conformer_original_pytorch_checkpoint_to_pytorch import (
    convert_espnet_state_dict_to_hf,
    remap_model_yaml_config,
)
from .convert_hifigan import load_weights, remap_hifigan_yaml_config


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")


def convert_FastSpeech2ConformerWithHifiGan_checkpoint(
    checkpoint_path,
    yaml_config_path,
    pytorch_dump_folder_path,
    repo_id=None,
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument(
        "--yaml_config_path", required=True, default=None, type=str, help="Path to config.yaml of model to convert"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        default=None,
        type=str,
        help="Path to the output `FastSpeech2ConformerModel` PyTorch model.",
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the Hugging Face hub."
    )

    args = parser.parse_args()

    convert_FastSpeech2ConformerWithHifiGan_checkpoint(
        args.checkpoint_path,
        args.yaml_config_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
