import argparse
import json
import os

import torch
from torch import nn

from transformers import NllbMoeConfig, NllbMoeModel
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME


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


def rename_fairseq_keys(state_dict, expert_idx=None):
    pass


def shard_on_the_fly(switch_checkpoint_path, dump_path, num_experts, dtype, weights_name: str = WEIGHTS_NAME):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nllb_moe_checkpoint_path",
        default="/home/arthur_huggingface_co/fairseq/weights/checkpoints/model_moe_54b/checkpoint_2_300000",
        type=str,
        required=False,
        help="Path to a directory containing a folder per layer. Follows the original Google format.",
    )
    parser.add_argument("--dtype", default="float32", type=str, required=False, help="dtype of the saved model")
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/home/arthur_huggingface_co/fairseq/weights/checkpoints/hf-converted-moe-54b",
        type=str,
        required=False,
        help="Path to the output pytorch model.",
    )
    args = parser.parse_args()
    metadata, index = shard_on_the_fly(
        args.nllb_moe_checkpoint_path,
        args.pytorch_dump_folder_path,
        128,
        args.dtype,
    )

    config = NllbMoeConfig.from_pretrained(
        "facebook/nllb-200-3.3B", encoder_sparse_step=4, decoder_sparse_step=4, num_experts=128
    )
    config.save_pretrained(args.pytorch_dump_folder_path)
    model = NllbMoeModel.from_pretrained(args.pytorch_dump_folder_path)
    print("Done")
    model.save_pretrained(args.pytorch_dump_folder_path)
