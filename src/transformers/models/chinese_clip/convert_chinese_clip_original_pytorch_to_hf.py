
import argparse

import torch

from transformers import ChineseCLIPConfig, ChineseCLIPModel


def copy_attn_layer(hf_attn_layer, pt_weights, prefix):
    pass


def copy_mlp(hf_mlp, pt_weights, prefix):
    pass


def copy_linear(hf_linear, pt_weights, prefix):
    pass


def copy_layer(hf_layer, pt_weights, prefix):
    pass


def copy_layers(hf_layers, pt_weights, prefix):
    pass


def copy_text_model_and_projection(hf_model, pt_weights):
    pass


def copy_vision_model_and_projection(hf_model, pt_weights):
    pass


@torch.no_grad()
def convert_chinese_clip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output folder storing converted hf PyTorch model.",
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to original github format ChineseCLIP checkpoint."
    )
    parser.add_argument(
        "--config_path", default=None, required=True, type=str, help="Path to hf config.json of model to convert."
    )
    args = parser.parse_args()

    convert_chinese_clip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
    print("The conversion is finished!")
