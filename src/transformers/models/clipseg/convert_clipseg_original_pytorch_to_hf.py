

import argparse
from io import BytesIO

import httpx
import torch
from PIL import Image

from transformers import (
    CLIPSegConfig,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
    CLIPSegTextConfig,
    CLIPSegVisionConfig,
    CLIPTokenizer,
    ViTImageProcessor,
)


def get_clipseg_config(model_name):
    pass


def rename_key(name):
    if "clip_model" in name:
        name = name.replace("clip_model", "clip")
    if "transformer" in name:
        if "visual" in name:
            name = name.replace("visual.transformer", "vision_model")
        else:
            name = name.replace("transformer", "text_model")
    if "resblocks" in name:
        name = name.replace("resblocks", "encoder.layers")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if "attn" in name and "self" not in name:
        name = name.replace("attn", "self_attn")
    if "token_embedding" in name:
        name = name.replace("token_embedding", "text_model.embeddings.token_embedding")
    if "positional_embedding" in name and "visual" not in name:
        name = name.replace("positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "ln_final" in name:
        name = name.replace("ln_final", "text_model.final_layer_norm")
    if "visual.class_embedding" in name:
        name = name.replace("visual.class_embedding", "vision_model.embeddings.class_embedding")
    if "visual.conv1" in name:
        name = name.replace("visual.conv1", "vision_model.embeddings.patch_embedding")
    if "visual.positional_embedding" in name:
        name = name.replace("visual.positional_embedding", "vision_model.embeddings.position_embedding.weight")
    if "visual.ln_pre" in name:
        name = name.replace("visual.ln_pre", "vision_model.pre_layrnorm")
    if "visual.ln_post" in name:
        name = name.replace("visual.ln_post", "vision_model.post_layernorm")
    if "visual.proj" in name:
        name = name.replace("visual.proj", "visual_projection.weight")
    if "text_projection" in name:
        name = name.replace("text_projection", "text_projection.weight")
    if "trans_conv" in name:
        name = name.replace("trans_conv", "transposed_convolution")
    if "film_mul" in name or "film_add" in name or "reduce" in name or "transposed_convolution" in name:
        name = "decoder." + name
    if "blocks" in name:
        name = name.replace("blocks", "decoder.layers")
    if "linear1" in name:
        name = name.replace("linear1", "mlp.fc1")
    if "linear2" in name:
        name = name.replace("linear2", "mlp.fc2")
    if "norm1" in name and "layer_" not in name:
        name = name.replace("norm1", "layer_norm1")
    if "norm2" in name and "layer_" not in name:
        name = name.replace("norm2", "layer_norm2")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        if key.startswith("clip_model") and "attn.in_proj" in key:
            key_split = key.split(".")
            if "visual" in key:
                layer_num = int(key_split[4])
                dim = config.vision_config.hidden_size
                prefix = "vision_model"
            else:
                layer_num = int(key_split[3])
                dim = config.text_config.hidden_size
                prefix = "text_model"

            if "weight" in key:
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[dim : dim * 2]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        elif "self_attn" in key and "out_proj" not in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            dim = config.reduce_dim
            if "weight" in key:
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[dim : dim * 2, :]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[dim : dim * 2]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        else:
            new_name = rename_key(key)
            if "visual_projection" in new_name or "text_projection" in new_name:
                val = val.T
            orig_state_dict[new_name] = val

    return orig_state_dict


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


def convert_clipseg_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="clipseg-rd64",
        type=str,
        choices=["clipseg-rd16", "clipseg-rd64", "clipseg-rd64-refined"],
        help=(
            "Name of the model. Supported models are: clipseg-rd64, clipseg-rd16 and clipseg-rd64-refined (rd meaning"
            " reduce dimension)"
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/CLIPSeg/clip_plus_rd64-uni.pth",
        type=str,
        help=(
            "Path to the original checkpoint. Note that the script assumes that the checkpoint includes both CLIP and"
            " the decoder weights."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_clipseg_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
