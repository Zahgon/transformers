
import argparse
import json

import gdown
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from transformers import TimesformerConfig, TimesformerForVideoClassification, VideoMAEImageProcessor


def get_timesformer_config(model_name):
    pass


def rename_key(name):
    if "encoder." in name:
        name = name.replace("encoder.", "")
    if "cls_token" in name:
        name = name.replace("cls_token", "timesformer.embeddings.cls_token")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "timesformer.embeddings.position_embeddings")
    if "time_embed" in name:
        name = name.replace("time_embed", "timesformer.embeddings.time_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "timesformer.embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "timesformer.embeddings.norm")
    if "blocks" in name:
        name = name.replace("blocks", "timesformer.encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name and "bias" not in name and "temporal" not in name:
        name = name.replace("attn", "attention.self")
    if "attn" in name and "temporal" not in name:
        name = name.replace("attn", "attention.attention")
    if "temporal_norm1" in name:
        name = name.replace("temporal_norm1", "temporal_layernorm")
    if "temporal_attn.proj" in name:
        name = name.replace("temporal_attn", "temporal_attention.output.dense")
    if "temporal_fc" in name:
        name = name.replace("temporal_fc", "temporal_dense")
    if "norm1" in name and "temporal" not in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "norm.weight" in name and "fc" not in name and "temporal" not in name:
        name = name.replace("norm.weight", "timesformer.layernorm.weight")
    if "norm.bias" in name and "fc" not in name and "temporal" not in name:
        name = name.replace("norm.bias", "timesformer.layernorm.bias")
    if "head" in name:
        name = name.replace("head", "classifier")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        if key.startswith("model."):
            key = key.replace("model.", "")

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            prefix = "timesformer.encoder.layer."
            if "temporal" in key:
                postfix = ".temporal_attention.attention.qkv."
            else:
                postfix = ".attention.attention.qkv."
            if "weight" in key:
                orig_state_dict[f"{prefix}{layer_num}{postfix}weight"] = val
            else:
                orig_state_dict[f"{prefix}{layer_num}{postfix}bias"] = val
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def prepare_video():
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)


def convert_timesformer_checkpoint(checkpoint_url, pytorch_dump_folder_path, model_name, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default="https://drive.google.com/u/1/uc?id=17yvuYp9L4mn-HpIcK5Zo6K3UoOy1kA5l&export=download",
        type=str,
        help=(
            "URL of the original PyTorch checkpoint (on Google Drive) you'd like to convert. Should be a direct"
            " download link."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--model_name", default="timesformer-base-finetuned-k400", type=str, help="Name of the model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_timesformer_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub
    )
