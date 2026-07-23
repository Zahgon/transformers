
import argparse
import os
import re

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from transformers import (
    DacModel,
    DiaConfig,
    DiaFeatureExtractor,
    DiaForConditionalGeneration,
    DiaProcessor,
    DiaTokenizer,
    GenerationConfig,
)
from transformers.utils.import_utils import is_tiktoken_available


shape_mappings = [
    "encoder.layers.*.mlp.gate_up_proj.weight",
    "encoder.layers.*.mlp.down_proj.weight",
    "encoder.layers.*.self_attention.q_proj.weight",
    "encoder.layers.*.self_attention.k_proj.weight",
    "encoder.layers.*.self_attention.v_proj.weight",
    "encoder.layers.*.self_attention.o_proj.weight",
    "decoder.layers.*.mlp.gate_up_proj.weight",
    "decoder.layers.*.mlp.down_proj.weight",
    "decoder.layers.*.self_attention.q_proj.weight",
    "decoder.layers.*.self_attention.k_proj.weight",
    "decoder.layers.*.self_attention.v_proj.weight",
    "decoder.layers.*.self_attention.o_proj.weight",
    "decoder.layers.*.cross_attention.q_proj.weight",
    "decoder.layers.*.cross_attention.k_proj.weight",
    "decoder.layers.*.cross_attention.v_proj.weight",
    "decoder.layers.*.cross_attention.o_proj.weight",
    "decoder.logits_dense.weight",
]

rename_mapping = {
    "mlp.wo": "mlp.down_proj",
    "mlp.wi_fused": "mlp.gate_up_proj",
}


def get_generation_config(config):
    pass


def convert_dia_model_to_hf(checkpoint_path, verbose=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, default="nari-labs/Dia-1.6B", help="Path to the downloaded checkpoints"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default="AntonV/Dia-1.6B", type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--convert_preprocessor",
        type=bool,
        default=True,
        help="Whether or not the preprocessor (tokenizer + feature extractor) should be converted along with the model.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Whether or not to log information during conversion.",
    )
    args = parser.parse_args()

    model = convert_dia_model_to_hf(args.checkpoint_path, args.verbose)
    if args.convert_preprocessor:
        try:
            if not is_tiktoken_available(with_blobfile=False):
                raise ModuleNotFoundError(
                    """`tiktoken` is not installed, use `pip install tiktoken` to convert the tokenizer"""
                )
        except Exception as e:
            print(e)
        else:
            processor = DiaProcessor(
                DiaFeatureExtractor(sampling_rate=44100, hop_length=512),
                DiaTokenizer(),
                DacModel.from_pretrained("descript/dac_44khz"),
            )
            processor.save_pretrained(args.pytorch_dump_folder_path)

    model.save_pretrained(args.pytorch_dump_folder_path)
    print(f"Saved converted checkpoint to {args.pytorch_dump_folder_path}")
