

import argparse
import glob
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    LlavaNextImageProcessor,
    LlavaNextVideoConfig,
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
    LlavaNextVideoVideoProcessor,
)


KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    ".vision_resampler": "",  # all lmms-lab models do avg pooling, so no vision_resampler
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "language_model.model.image_newline": "image_newline",
}

chat_vicuna = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'][0]['text'] }}"
    "{% else %}"
    "{{ message['role'].upper() + ': '}}"
    "{% endif %}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] + ' '}}"
    "{% endfor %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'ASSISTANT:' }}"
    "{% endif %}"
)

chat_mistral = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' }}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] }}"
    "{% endfor %}"
    "{{' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    r"{{ ' ' + message['content'][0]['text'] + '<\s> '}}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
)

chat_yi = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] }}"
    "{% endfor %}"
    "{{'<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

model2template = {
    "lmms-lab/LLaVA-NeXT-Video-7B-32K": chat_mistral,
    "lmms-lab/LLaVA-NeXT-Video-7B": chat_vicuna,
    "lmms-lab/LLaVA-NeXT-Video-7B-DPO": chat_vicuna,
    "lmms-lab/LLaVA-NeXT-Video-34B": chat_yi,
    "lmms-lab/LLaVA-NeXT-Video-34B-DPO": chat_yi,
}


def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value.to(torch.bfloat16)
    return new_state_dict


def convert_llava_to_hf(model_id, pytorch_dump_folder_path, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="lmms-lab/LLaVA-NeXT-Video-7B",
        choices=[
            "lmms-lab/LLaVA-NeXT-Video-7B",
            "lmms-lab/LLaVA-NeXT-Video-7B-DPO",
            "lmms-lab/LLaVA-NeXT-Video-7B-32K",
            "lmms-lab/LLaVA-NeXT-Video-34B",
            "lmms-lab/LLaVA-NeXT-Video-34B-DPO",
        ],
        required=False,
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

    convert_llava_to_hf(args.model_id, args.pytorch_dump_folder_path, args.push_to_hub)
