
import argparse

import torch
from torch import nn

from transformers import MBartConfig, MBartForConditionalGeneration


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    pass


def convert_fairseq_mbart_checkpoint_from_disk(
    checkpoint_path, hf_config_path="facebook/mbart-large-en-ro", finetuned=False, mbart_50=False
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config",
        default="facebook/mbart-large-cc25",
        type=str,
        help="Which huggingface architecture to use: mbart-large",
    )
    parser.add_argument("--mbart_50", action="store_true", help="whether the model is mMART-50 checkpoint")
    parser.add_argument("--finetuned", action="store_true", help="whether the model is a fine-tuned checkpoint")
    args = parser.parse_args()
    model = convert_fairseq_mbart_checkpoint_from_disk(
        args.fairseq_path, hf_config_path=args.hf_config, finetuned=args.finetuned, mbart_50=args.mbart_50
    )
    model.save_pretrained(args.pytorch_dump_folder_path)
