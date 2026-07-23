
import argparse
import json
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_audio_spectrogram_transformer_config(model_name):
    pass


def rename_key(name):
    if "module.v" in name:
        name = name.replace("module.v", "audio_spectrogram_transformer")
    if "cls_token" in name:
        name = name.replace("cls_token", "embeddings.cls_token")
    if "dist_token" in name:
        name = name.replace("dist_token", "embeddings.distillation_token")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "embeddings.position_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "blocks" in name:
        name = name.replace("blocks", "encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "audio_spectrogram_transformer.norm" in name:
        name = name.replace("audio_spectrogram_transformer.norm", "audio_spectrogram_transformer.layernorm")
    if "module.mlp_head.0" in name:
        name = name.replace("module.mlp_head.0", "classifier.layernorm")
    if "module.mlp_head.1" in name:
        name = name.replace("module.mlp_head.1", "classifier.dense")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[3])
            dim = config.hidden_size
            if "weight" in key:
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.query.weight"
                ] = val[:dim, :]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.key.weight"
                ] = val[dim : dim * 2, :]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.value.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.query.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.key.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"audio_spectrogram_transformer.encoder.layer.{layer_num}.attention.attention.value.bias"
                ] = val[-dim:]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def remove_keys(state_dict):
    pass


@torch.no_grad()
def convert_audio_spectrogram_transformer_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="ast-finetuned-audioset-10-10-0.4593",
        type=str,
        help="Name of the Audio Spectrogram Transformer model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_audio_spectrogram_transformer_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
