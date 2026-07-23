
import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml

from transformers import (
    FastSpeech2ConformerConfig,
    FastSpeech2ConformerModel,
    FastSpeech2ConformerTokenizer,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")

CONFIG_MAPPING = {
    "adim": "hidden_size",
    "aheads": "num_attention_heads",
    "conformer_dec_kernel_size": "decoder_kernel_size",
    "conformer_enc_kernel_size": "encoder_kernel_size",
    "decoder_normalize_before": "decoder_normalize_before",
    "dlayers": "decoder_layers",
    "dunits": "decoder_linear_units",
    "duration_predictor_chans": "duration_predictor_channels",
    "duration_predictor_kernel_size": "duration_predictor_kernel_size",
    "duration_predictor_layers": "duration_predictor_layers",
    "elayers": "encoder_layers",
    "encoder_normalize_before": "encoder_normalize_before",
    "energy_embed_dropout": "energy_embed_dropout",
    "energy_embed_kernel_size": "energy_embed_kernel_size",
    "energy_predictor_chans": "energy_predictor_channels",
    "energy_predictor_dropout": "energy_predictor_dropout",
    "energy_predictor_kernel_size": "energy_predictor_kernel_size",
    "energy_predictor_layers": "energy_predictor_layers",
    "eunits": "encoder_linear_units",
    "pitch_embed_dropout": "pitch_embed_dropout",
    "pitch_embed_kernel_size": "pitch_embed_kernel_size",
    "pitch_predictor_chans": "pitch_predictor_channels",
    "pitch_predictor_dropout": "pitch_predictor_dropout",
    "pitch_predictor_kernel_size": "pitch_predictor_kernel_size",
    "pitch_predictor_layers": "pitch_predictor_layers",
    "positionwise_conv_kernel_size": "positionwise_conv_kernel_size",
    "postnet_chans": "speech_decoder_postnet_units",
    "postnet_filts": "speech_decoder_postnet_kernel",
    "postnet_layers": "speech_decoder_postnet_layers",
    "reduction_factor": "reduction_factor",
    "stop_gradient_from_energy_predictor": "stop_gradient_from_energy_predictor",
    "stop_gradient_from_pitch_predictor": "stop_gradient_from_pitch_predictor",
    "transformer_dec_attn_dropout_rate": "decoder_attention_dropout_rate",
    "transformer_dec_dropout_rate": "decoder_dropout_rate",
    "transformer_dec_positional_dropout_rate": "decoder_positional_dropout_rate",
    "transformer_enc_attn_dropout_rate": "encoder_attention_dropout_rate",
    "transformer_enc_dropout_rate": "encoder_dropout_rate",
    "transformer_enc_positional_dropout_rate": "encoder_positional_dropout_rate",
    "use_cnn_in_conformer": "use_cnn_in_conformer",
    "use_macaron_style_in_conformer": "use_macaron_style_in_conformer",
    "use_masking": "use_masking",
    "use_weighted_masking": "use_weighted_masking",
    "idim": "input_dim",
    "odim": "num_mel_bins",
    "spk_embed_dim": "speaker_embed_dim",
    "langs": "num_languages",
    "spks": "num_speakers",
}


def remap_model_yaml_config(yaml_config_path):
    pass


def convert_espnet_state_dict_to_hf(state_dict):
    pass


@torch.no_grad()
def convert_FastSpeech2ConformerModel_checkpoint(
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
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the Hugging Face hub."
    )

    args = parser.parse_args()
    convert_FastSpeech2ConformerModel_checkpoint(
        args.checkpoint_path,
        args.yaml_config_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
