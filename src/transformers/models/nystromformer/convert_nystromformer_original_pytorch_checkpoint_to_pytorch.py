

import argparse

import torch

from transformers import NystromformerConfig, NystromformerForMaskedLM


def rename_key(orig_key):
    if "model" in orig_key:
        orig_key = orig_key.replace("model.", "")
    if "norm1" in orig_key:
        orig_key = orig_key.replace("norm1", "attention.output.LayerNorm")
    if "norm2" in orig_key:
        orig_key = orig_key.replace("norm2", "output.LayerNorm")
    if "norm" in orig_key:
        orig_key = orig_key.replace("norm", "LayerNorm")
    if "transformer" in orig_key:
        layer_num = orig_key.split(".")[0].split("_")[-1]
        orig_key = orig_key.replace(f"transformer_{layer_num}", f"encoder.layer.{layer_num}")
    if "mha.attn" in orig_key:
        orig_key = orig_key.replace("mha.attn", "attention.self")
    if "mha" in orig_key:
        orig_key = orig_key.replace("mha", "attention")
    if "W_q" in orig_key:
        orig_key = orig_key.replace("W_q", "self.query")
    if "W_k" in orig_key:
        orig_key = orig_key.replace("W_k", "self.key")
    if "W_v" in orig_key:
        orig_key = orig_key.replace("W_v", "self.value")
    if "ff1" in orig_key:
        orig_key = orig_key.replace("ff1", "intermediate.dense")
    if "ff2" in orig_key:
        orig_key = orig_key.replace("ff2", "output.dense")
    if "ff" in orig_key:
        orig_key = orig_key.replace("ff", "output.dense")
    if "mlm_class" in orig_key:
        orig_key = orig_key.replace("mlm.mlm_class", "cls.predictions.decoder")
    if "mlm" in orig_key:
        orig_key = orig_key.replace("mlm", "cls.predictions.transform")
    if "cls" not in orig_key:
        orig_key = "nystromformer." + orig_key

    return orig_key


def convert_checkpoint_helper(config, orig_state_dict):
    pass


def convert_nystromformer_checkpoint(checkpoint_path, nystromformer_config_file, pytorch_dump_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_model_path", default=None, type=str, required=True, help="Path to Nystromformer pytorch checkpoint."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for Nystromformer model config.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_nystromformer_checkpoint(args.pytorch_model_path, args.config_file, args.pytorch_dump_path)
