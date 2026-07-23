

import argparse
import gc
import os

import safetensors
import torch
from huggingface_hub import hf_hub_download

from transformers import PhiConfig, PhiForCausalLM


_MODELS = {
    "microsoft/phi-1": ["https://huggingface.co/microsoft/phi-1/blob/main/pytorch_model.bin"],
    "microsoft/phi-1_5": ["https://huggingface.co/microsoft/phi-1_5/blob/main/pytorch_model.bin"],
    "microsoft/phi-2": [
        "https://huggingface.co/microsoft/phi-2/blob/main/model-00001-of-00002.safetensors",
        "https://huggingface.co/microsoft/phi-2/blob/main/model-00002-of-00002.safetensors",
    ],
}

PHI_MAPPING = {
    "transformer.embd.wte.weight": "model.embed_tokens.weight",
    "lm_head.linear": "lm_head",
    "lm_head.ln": "model.final_layernorm",
    "layers": "model.layers",
    "transformer": "model",
    ".h.": ".layers.",
    "ln": "input_layernorm",
    "mixer": "self_attn",
    "Wqkv": "query_key_value",
    "out_proj": "dense",
}


def convert_weights(original_weights, mapping, config):
    converted_weights = {}
    original_weights_keys = sorted(original_weights.keys())

    for original_weights_key in original_weights_keys:
        new_key = original_weights_key

        if "rotary_emb" in new_key:
            continue

        if "Wqkv" in new_key:
            if "weight" in new_key:
                weight = original_weights[new_key]
                weights_shape = weight.shape
                weight = (
                    weight.view(3, config.num_attention_heads, -1, config.hidden_size)
                    .transpose(0, 1)
                    .reshape(*weights_shape)
                )
                original_weights[new_key] = weight
            elif "bias" in new_key:
                bias = original_weights[new_key]
                bias_shape = bias.shape
                bias = bias.view(3, config.num_attention_heads, -1).transpose(0, 1).reshape(*bias_shape)
                original_weights[new_key] = bias

        for k, v in mapping.items():
            if k in new_key:
                new_key = new_key.replace(k, v)

        converted_weights[new_key] = original_weights.pop(original_weights_key)

    return converted_weights


def _download(url: str, root: str):
    repo_id = f"{url.split('/')[3]}/{url.split('/')[4]}"
    filename = f"{url.split('/')[-1]}"
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_filename=root,
        local_dir_use_symlinks=False,
    )


def convert_phi_weights(
    model_name, checkpoint_path, pytorch_dump_folder_path, use_cuda, save_weights_directly, _MODELS
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to convert. (Please enter one of the following: phi-1, phi-1_5, phi-2). If nothing is provided, all models will be converted.",
        default=None,
    )
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to the folder of downloaded checkpoints. (Please enter full path)"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model. (Please enter full path)",
    )
    parser.add_argument(
        "--use_cuda",
        default=False,
        type=bool,
        help="Whether to load the weights on GPU during conversion or not, False by default",
    )
    parser.add_argument(
        "--save_weights_directly",
        default=True,
        type=bool,
        help="Whether to save the weights directly after conversion or load the weight to the Phi model and then save "
        "the Phi model along with weights. True by default",
    )

    args = parser.parse_args()
    convert_phi_weights(
        args.model_name,
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.use_cuda,
        args.save_weights_directly,
        _MODELS,
    )
