
import argparse

import torch
from torch import nn

from transformers import Speech2TextConfig, Speech2TextForConditionalGeneration


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "decoder.output_projection.weight",
        "_float_tensor",
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        if "transformer_layers" in key:
            s_dict[key.replace("transformer_layers", "layers")] = s_dict.pop(key)
        elif "subsample" in key:
            s_dict[key.replace("subsample", "conv")] = s_dict.pop(key)


def make_linear_from_emb(emb):
    pass


def convert_fairseq_s2t_checkpoint_to_tfms(checkpoint_path, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairseq_path", type=str, help="Path to the fairseq model (.pt) file.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_fairseq_s2t_checkpoint_to_tfms(args.fairseq_path, args.pytorch_dump_folder_path)
