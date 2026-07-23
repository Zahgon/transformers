
import argparse
import json

import gdown
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from transformers import (
    VideoMAEConfig,
    VideoMAEForPreTraining,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)


def get_videomae_config(model_name):
    pass


def set_architecture_configs(model_name, config):
    pass


def rename_key(name):
    if "encoder." in name:
        name = name.replace("encoder.", "")
    if "cls_token" in name:
        name = name.replace("cls_token", "videomae.embeddings.cls_token")
    if "decoder_pos_embed" in name:
        name = name.replace("decoder_pos_embed", "decoder.decoder_pos_embed")
    if "pos_embed" in name and "decoder" not in name:
        name = name.replace("pos_embed", "videomae.embeddings.position_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "videomae.embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "videomae.embeddings.norm")
    if "decoder.blocks" in name:
        name = name.replace("decoder.blocks", "decoder.decoder_layers")
    if "blocks" in name:
        name = name.replace("blocks", "videomae.encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name and "bias" not in name:
        name = name.replace("attn", "attention.self")
    if "attn" in name:
        name = name.replace("attn", "attention.attention")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "decoder_embed" in name:
        name = name.replace("decoder_embed", "decoder.decoder_embed")
    if "decoder_norm" in name:
        name = name.replace("decoder_norm", "decoder.decoder_norm")
    if "decoder_pred" in name:
        name = name.replace("decoder_pred", "decoder.decoder_pred")
    if "norm.weight" in name and "decoder" not in name and "fc" not in name:
        name = name.replace("norm.weight", "videomae.layernorm.weight")
    if "norm.bias" in name and "decoder" not in name and "fc" not in name:
        name = name.replace("norm.bias", "videomae.layernorm.bias")
    if "head" in name and "decoder" not in name:
        name = name.replace("head", "classifier")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        if key.startswith("encoder."):
            key = key.replace("encoder.", "")

        if "qkv" in key:
            key_split = key.split(".")
            if key.startswith("decoder.blocks"):
                dim = config.decoder_hidden_size
                layer_num = int(key_split[2])
                prefix = "decoder.decoder_layers."
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
            else:
                dim = config.hidden_size
                layer_num = int(key_split[1])
                prefix = "videomae.encoder.layer."
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def prepare_video():
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)


def convert_videomae_checkpoint(checkpoint_url, pytorch_dump_folder_path, model_name, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default="https://drive.google.com/u/1/uc?id=1tEhLyskjb755TJ65ptsrafUG2llSwQE1&amp;export=download&amp;confirm=t&amp;uuid=aa3276eb-fb7e-482a-adec-dc7171df14c4",
        type=str,
        help=(
            "URL of the original PyTorch checkpoint (on Google Drive) you'd like to convert. Should be a direct"
            " download link."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/Users/nielsrogge/Documents/VideoMAE/Test",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--model_name", default="videomae-base", type=str, help="Name of the model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_videomae_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub)
