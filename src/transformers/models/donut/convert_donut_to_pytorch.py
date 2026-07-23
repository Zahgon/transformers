
import argparse

import torch
from datasets import load_dataset
from donut import DonutModel

from transformers import (
    DonutImageProcessor,
    DonutProcessor,
    DonutSwinConfig,
    DonutSwinModel,
    MBartConfig,
    MBartForCausalLM,
    VisionEncoderDecoderModel,
    XLMRobertaTokenizerFast,
)


def get_configs(model):
    pass


def rename_key(name):
    if "encoder.model" in name:
        name = name.replace("encoder.model", "encoder")
    if "decoder.model" in name:
        name = name.replace("decoder.model", "decoder")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    if name.startswith("encoder"):
        if "layers" in name:
            name = "encoder." + name
        if "attn.proj" in name:
            name = name.replace("attn.proj", "attention.output.dense")
        if "attn" in name and "mask" not in name:
            name = name.replace("attn", "attention.self")
        if "norm1" in name:
            name = name.replace("norm1", "layernorm_before")
        if "norm2" in name:
            name = name.replace("norm2", "layernorm_after")
        if "mlp.fc1" in name:
            name = name.replace("mlp.fc1", "intermediate.dense")
        if "mlp.fc2" in name:
            name = name.replace("mlp.fc2", "output.dense")

        if name == "encoder.norm.weight":
            name = "encoder.layernorm.weight"
        if name == "encoder.norm.bias":
            name = "encoder.layernorm.bias"

    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[3])
            block_num = int(key_split[5])
            dim = model.encoder.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            if "weight" in key:
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"] = (
                    val[dim : dim * 2, :]
                )
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"] = (
                    val[:dim]
                )
                orig_state_dict[f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"] = (
                    val[dim : dim * 2]
                )
                orig_state_dict[f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"] = (
                    val[-dim:]
                )
        elif "attn_mask" in key or key in ["encoder.model.norm.weight", "encoder.model.norm.bias"]:
            pass
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_donut_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="naver-clova-ix/donut-base-finetuned-docvqa",
        required=False,
        type=str,
        help="Name of the original model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_donut_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
