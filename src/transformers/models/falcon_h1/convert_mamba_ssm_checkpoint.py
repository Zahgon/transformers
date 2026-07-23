
import argparse

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, FalconH1Config, FalconH1ForCausalLM


CONVERSION_MAPPING = {
    "backbone": "model",
    "embeddings": "embed_tokens",
    "mixer.": "",
    "mixer_ssm": "mamba",
    "mixer_attn": "self_attn",
    "mlp.": "feed_forward.",
    "mlp_norm": "pre_ff_layernorm",
    "ssm_proj": "mamba.in_proj",
    "attn_out_proj": "o_proj",
    ".norm.": ".input_layernorm.",
    ".mamba.input_layernorm.": ".mamba.norm.",
    ".ssm_out_proj.": ".mamba.out_proj.",
    "norm_f": "final_layernorm",
}


def convert_falcon_h1_to_hf(input_model_path, output_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--mamba_ssm_checkpoint_directory",
        type=str,
        required=True,
        help="Path to a directory containing the `pytorch_model.bin` mamba_ssm checkpoint file to be converted.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to directory to save the converted output model to."
    )
    args = parser.parse_args()

    convert_falcon_h1_to_hf(
        args.mamba_ssm_checkpoint_directory,
        args.output_dir,
    )
