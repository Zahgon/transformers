

import argparse
import gc
import glob
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    LlavaOnevisionConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionImageProcessor,
    LlavaOnevisionProcessor,
    LlavaOnevisionVideoProcessor,
    SiglipVisionConfig,
)


KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "language_model.model.image_newline": "image_newline",
}

chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all video then #}{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}{{ '<video>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] }}{% endgeneration %}{% endfor %}{% endif %}{{'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    if "lm_head.weight" not in original_state_dict:
        original_state_dict["lm_head.weight"] = original_state_dict["model.embed_tokens.weight"].clone()

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value.to(torch.float16)
    return new_state_dict


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


def convert_llava_to_hf(model_id, pytorch_dump_folder_path, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="lmms-lab/llava-onevision-qwen2-0.5b-ov",
        choices=[
            "lmms-lab/llava-onevision-qwen2-0.5b-ov",
            "lmms-lab/llava-onevision-qwen2-0.5b-si",
            "lmms-lab/llava-onevision-qwen2-7b-si",
            "lmms-lab/llava-onevision-qwen2-7b-ov",
            "lmms-lab/llava-onevision-qwen2-72b-si",
            "lmms-lab/llava-onevision-qwen2-72b-ov",
            "lmms-lab/llava-onevision-qwen2-7b-ov-chat",
            "lmms-lab/llava-onevision-qwen2-72b-ov-chat",
        ],
        required=False,
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, required=True, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )
    args = parser.parse_args()

    convert_llava_to_hf(args.model_id, args.pytorch_dump_folder_path, args.push_to_hub)
