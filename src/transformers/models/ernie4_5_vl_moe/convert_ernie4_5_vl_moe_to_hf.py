
import argparse
import json
import os
from pathlib import Path
from shutil import copyfile

from huggingface_hub import hf_hub_download, snapshot_download
from tokenizers import AddedToken

from transformers import (
    AutoTokenizer,
    Ernie4_5_VLMoeConfig,
    Ernie4_5_VLMoeImageProcessorFast,
    Ernie4_5_VLMoeProcessor,
    Ernie4_5_VLMoeVideoProcessor,
    LlamaTokenizer,
)


CONFIG_NAME = "config.json"
VALID_VISION_CONFIG_KEYS = [
    "depth",
    "hidden_size",
    "hidden_act",
    "num_heads",
    "in_channels",
    "patch_size",
    "spatial_merge_size",
]
VALID_TEXT_CONFIG_KEYS = [
    "hidden_size",
    "intermediate_size",
    "max_position_embeddings",
    "moe_intermediate_size",
    "moe_k",
    "moe_layer_interval",
    "moe_num_shared_experts",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_theta",
    "vocab_size",
    "tie_word_embeddings",
    "use_cache",
    "use_bias",
]
TEXT_TO_VISION_CONFIG_KEYS = [
    "spatial_conv_size",
    "temporal_conv_size",
]
ALL_VISION_CONFIG_KEYS = VALID_VISION_CONFIG_KEYS + TEXT_TO_VISION_CONFIG_KEYS + ["intermediate_size"]
ALL_TEXT_CONFIG_KEYS = VALID_TEXT_CONFIG_KEYS + [
    "hidden_act",
    "mlp_layer_types",
    "moe_num_experts",
    "rope_parameters",
]

TMP_TOKENIZER_DIR = "/tmp/ernie_vl_tokenizer"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
DEFAULT_CHAT_TEMPLATE = """
{%- set image_count = namespace(value=0) -%}
{%- set video_count = namespace(value=0) -%}
{{- '<|begin_of_sentence|>' }}
{%- for message in messages -%}
    {%- if message.role in ['system', 'user'] -%}
        {%- if message.role == 'user' -%}
            {{- 'User: ' -}}
        {%- endif -%}
        {%- if message.content is string -%}
            {{- message.content -}}
        {%- else -%}
            {%- for content_item in message.content -%}
                {%- if content_item.type == 'text' -%}
                    {{- content_item.text -}}
                {%- elif content_item.type in ['image_url', 'image'] -%}
                    {%- set image_count.value = image_count.value + 1 -%}
                    Picture {{ image_count.value }}:<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>
                {%- elif content_item.type in ['video_url', 'video'] -%}
                    {%- set video_count.value = video_count.value + 1 -%}
                    Video {{ video_count.value }}:<|VIDEO_START|><|VIDEO_PLACEHOLDER|><|VIDEO_END|>
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
        {%- if message.role == 'system' -%}
            {{- '
            ' -}}
        {%- endif -%}
    {%- elif message.role == 'assistant' -%}
        {%- macro extract_text_content(content_field) -%}
            {%- if content_field is string -%}
                {{- content_field -}}
            {%- elif content_field is iterable and content_field is not string -%}
                {%- set ns = namespace(text_parts=[]) -%}
                {%- set text_parts = [] -%}
                {%- for item in content_field -%}
                    {%- if item.type == 'text' -%}
                        {%- set ns.text_parts = ns.text_parts + [item.text] -%}
                    {%- endif -%}
                {%- endfor -%}
                {{- ns.text_parts | join("") -}}
            {%- else -%}
                {{- '' -}}
            {%- endif -%}
        {%- endmacro -%}
        {%- set reasoning_content = extract_text_content(message.reasoning_content) -%}
        {%- set content = extract_text_content(message.content) -%}
        {%- if '</think>' in content %}
            {%- set reasoning_content = content.split('</think>')[0].rstrip('
                        ').split('<think>')[-1].lstrip('
                        ') %}
            {%- set content = content.split('</think>')[-1].lstrip('
                        ') %}
        {%- endif %}
        {%- if reasoning_content %}
            {{- '
            ' + 'Assistant: ' + '<think>
            ' + reasoning_content.strip('
                        ') + '
            </think>
            ' + content.lstrip('
            ') }}
        {%- else %}
            {{- '
            ' + 'Assistant: ' + content }}
        {%- endif %}
        {{- '<|end_of_sentence |>' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt is not defined or add_generation_prompt is true %}
    {{- '\nAssistant: ' -}}
    {%- if (enable_thinking is defined and enable_thinking is false) or enable_thinking is not defined %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
    {%- if enable_thinking is defined and enable_thinking is true %}{{- '<think>' }}{%- endif %}
{%- endif %}"""
FONT_REPO = "AntonV/ernie4_5_fonts"
FONT_NAME = "Roboto-Regular.ttf"


def load_json(save_dir, filename):
    with open(os.path.join(save_dir, filename), "r") as f:
        return json.load(f)


def write_json(json_object, save_dir, filename):
    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump(json_object, f, indent=2, sort_keys=True, ensure_ascii=False)


def convert_vision_config_to_hf(vision_config, original_config, original_vision_config):
    for key in VALID_VISION_CONFIG_KEYS:
        vision_config[key] = original_vision_config[key]
    vision_config["intermediate_size"] = original_vision_config["hidden_size"] * original_vision_config["mlp_ratio"]

    for key in TEXT_TO_VISION_CONFIG_KEYS:
        vision_config[key.replace("conv", "merge")] = original_config[key]
    vision_config["rms_norm_eps"] = 1e-6

    for key in list(vision_config.keys()):
        if key not in ALL_VISION_CONFIG_KEYS:
            del vision_config[key]

    return vision_config


def convert_text_config_to_hf(text_config, original_config):
    for key in VALID_TEXT_CONFIG_KEYS:
        text_config[key] = original_config.get(key)

    text_config["hidden_act"] = "silu"  # default value which is not explicit in their json
    text_config["use_cache"] = True  # not always included but we should default to `True`
    text_config["moe_num_experts"] = original_config["moe_num_experts"][0]  # the same for both modalities
    text_config["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": 500_000.0,
        "mrope_section": [22, 22, 20],
    }
    if text_config["moe_num_shared_experts"] is None:
        text_config["moe_num_shared_experts"] = 0

    text_config["mlp_layer_types"] = []
    for layer_idx in range(text_config["num_hidden_layers"]):
        if (
            ((layer_idx + 1) % text_config["moe_layer_interval"] == 0)
            and layer_idx >= min(original_config["moe_layer_start_index"])
            and layer_idx <= max(original_config["moe_layer_end_index"])
        ):
            text_config["mlp_layer_types"].append("sparse")
        else:
            text_config["mlp_layer_types"].append("dense")
    text_config.pop("moe_layer_interval", None)

    for key in list(text_config.keys()):
        if key not in ALL_TEXT_CONFIG_KEYS:
            del text_config[key]

    return text_config


def convert_config(model_path, save_dir):
    checkpoint_path = snapshot_download(repo_id=model_path, allow_patterns=["*config*"])
    for filename in sorted(os.listdir(checkpoint_path)):
        if filename == CONFIG_NAME:
            hf_config = Ernie4_5_VLMoeConfig()
            original_config = load_json(checkpoint_path, filename)

            image_token_id = original_config["im_patch_id"]

            vision_config = hf_config.vision_config.to_dict()
            original_vision_config = original_config["vision_config"]
            vision_config = convert_vision_config_to_hf(vision_config, original_config, original_vision_config)

            text_config = hf_config.text_config.to_dict()
            text_config = convert_text_config_to_hf(text_config, original_config)

            final_config = Ernie4_5_VLMoeConfig(
                text_config=text_config,
                vision_config=vision_config,
                image_token_id=image_token_id,
            )
            setattr(final_config, "architectures", original_config["architectures"])  # carry over

            final_config.save_pretrained(save_dir)
            break
    print("Converted model config\n")


def convert_tokenizer(original_tokenizer_path, save_dir):
    pass


def convert_processor(model_path, save_dir):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="baidu/ERNIE-4.5-VL-28B-A3B-PT",
        help="Path to the downloaded checkpoint",
    )
    parser.add_argument("--output_folder", default="AntonV/ErnieVL", type=str, help="Path to your output directory.")
    parser.add_argument(
        "--convert_preprocessor",
        type=bool,
        default=True,
        help="Whether or not the preprocessor (tokenizer + image/video processors) should be converted along with the model.",
    )
    args = parser.parse_args()

    convert_config(args.checkpoint_path, args.output_folder)
    if args.convert_preprocessor:
        convert_processor(args.checkpoint_path, args.output_folder)

    print(f"Saved converted checkpoint to {args.output_folder}")
