
import argparse

import torch
from clip import load

from transformers import CLIPConfig, CLIPModel


def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    pass


def copy_mlp(hf_mlp, pt_mlp):
    pass


def copy_linear(hf_linear, pt_linear):
    pass


def copy_layer(hf_layer, pt_layer):
    pass


def copy_layers(hf_layers, pt_layers):
    pass


def copy_encoder(hf_encoder, pt_model):
    pass


def copy_text_model_and_projection(hf_model, pt_model):
    pass


def copy_vison_model_and_projection(hf_model, pt_model):
    pass


@torch.no_grad()
def convert_clip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to OpenAI checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    convert_clip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
