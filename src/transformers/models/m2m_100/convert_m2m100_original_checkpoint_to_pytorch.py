
import argparse

import torch
from torch import nn

from transformers import M2M100Config, M2M100ForConditionalGeneration


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


def make_linear_from_emb(emb):
    pass


def convert_fairseq_m2m100_checkpoint_from_disk(checkpoint_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fairseq_path", type=str, help="path to a model.pt on local filesystem.")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    model = convert_fairseq_m2m100_checkpoint_from_disk(args.fairseq_pathß)
    model.save_pretrained(args.pytorch_dump_folder_path)
