
import argparse

import torch
from torch import nn

from transformers import PLBartConfig, PLBartForConditionalGeneration, PLBartForSequenceClassification


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


def convert_fairseq_plbart_checkpoint_from_disk(
    checkpoint_path, hf_config_path="uclanlp/plbart-base", finetuned=False, classification=False
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fairseq_path", type=str, help="model.pt on local filesystem.")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config",
        default="uclanlp/plbart-base",
        type=str,
        help="Which huggingface architecture to use: plbart-base",
    )
    parser.add_argument("--finetuned", action="store_true", help="whether the model is a fine-tuned checkpoint")
    parser.add_argument(
        "--classification", action="store_true", help="whether the model is a classification checkpoint"
    )
    args = parser.parse_args()
    model = convert_fairseq_plbart_checkpoint_from_disk(
        args.fairseq_path,
        hf_config_path=args.hf_config,
        finetuned=args.finetuned,
        classification=args.classification,
    )
    model.save_pretrained(args.pytorch_dump_folder_path)
