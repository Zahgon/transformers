
import argparse

import gdown
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from transformers import (
    CLIPTokenizer,
    CLIPTokenizerFast,
    VideoMAEImageProcessor,
    XCLIPConfig,
    XCLIPModel,
    XCLIPProcessor,
    XCLIPTextConfig,
    XCLIPVisionConfig,
)


def get_xclip_config(model_name, num_frames):
    pass


def rename_key(name):
    if name == "token_embedding.weight":
        name = name.replace("token_embedding.weight", "text_model.embeddings.token_embedding.weight")
    if name == "positional_embedding":
        name = name.replace("positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if name.startswith("transformer.resblocks"):
        name = name.replace("transformer.resblocks", "text_model.encoder.layers")
    if "attn.out_proj" in name and "message" not in name:
        name = name.replace("attn.out_proj", "self_attn.out_proj")
    if "ln_final" in name:
        name = name.replace("ln_final", "text_model.final_layer_norm")
    if name == "visual.class_embedding":
        name = name.replace("visual.class_embedding", "vision_model.embeddings.class_embedding")
    if name == "visual.positional_embedding":
        name = name.replace("visual.positional_embedding", "vision_model.embeddings.position_embedding.weight")
    if name.startswith("visual.transformer.resblocks"):
        name = name.replace("visual.transformer.resblocks", "vision_model.encoder.layers")
    if "visual.conv1" in name:
        name = name.replace("visual.conv1", "vision_model.embeddings.patch_embedding")
    if "visual.ln_pre" in name:
        name = name.replace("visual.ln_pre", "vision_model.pre_layernorm")
    if "visual.ln_post" in name:
        name = name.replace("visual.ln_post", "vision_model.post_layernorm")
    if "visual.proj" in name:
        name = name.replace("visual.proj", "visual_projection.weight")
    if "text_projection" in name:
        name = name.replace("text_projection", "text_projection.weight")
    if "prompts_visual_proj" in name:
        name = name.replace("prompts_visual_proj", "prompts_visual_projection")
    if "prompts_visual_ln" in name:
        name = name.replace("prompts_visual_ln", "prompts_visual_layernorm")
    if name == "mit.positional_embedding":
        name = name.replace("positional", "position")
    if name.startswith("mit.resblocks"):
        name = name.replace("mit.resblocks", "mit.encoder.layers")
    if name.startswith("prompts_generator.norm"):
        name = name.replace("prompts_generator.norm", "prompts_generator.layernorm")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        if "attn.in_proj" in key:
            key_split = key.split(".")
            if key.startswith("visual"):
                layer_num = key_split[3]
                dim = config.vision_config.hidden_size
                if "message_attn" in key:
                    if "weight" in key:
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.q_proj.weight"] = val[
                            :dim, :
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.k_proj.weight"] = val[
                            dim : dim * 2, :
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.v_proj.weight"] = val[
                            -dim:, :
                        ]
                    else:
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.q_proj.bias"] = val[
                            :dim
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.k_proj.bias"] = val[
                            dim : dim * 2
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.message_attn.v_proj.bias"] = val[
                            -dim:
                        ]
                else:
                    if "weight" in key:
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[
                            :dim, :
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[
                            dim : dim * 2, :
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[
                            -dim:, :
                        ]
                    else:
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[
                            dim : dim * 2
                        ]
                        orig_state_dict[f"vision_model.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
            elif key.startswith("mit"):
                layer_num = key_split[2]
                dim = config.vision_config.mit_hidden_size
                if "weight" in key:
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
                else:
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[dim : dim * 2]
                    orig_state_dict[f"mit.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
            else:
                layer_num = key_split[2]
                dim = config.text_config.hidden_size
                if "weight" in key:
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[
                        dim : dim * 2, :
                    ]
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
                else:
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[
                        dim : dim * 2
                    ]
                    orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        else:
            new_key_name = rename_key(key)
            if new_key_name in ["visual_projection.weight", "text_projection.weight"]:
                val = val.T
            orig_state_dict[new_key_name] = val

    return orig_state_dict


def prepare_video(num_frames):
    if num_frames == 8:
        filename = "eating_spaghetti_8_frames.npy"
    elif num_frames == 16:
        filename = "eating_spaghetti.npy"
    elif num_frames == 32:
        filename = "eating_spaghetti_32_frames.npy"
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video",
        filename=filename,
        repo_type="dataset",
    )
    video = np.load(file)
    return list(video)


def convert_xclip_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="xclip-base-patch32",
        type=str,
        help="Name of the model.",
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
    convert_xclip_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
