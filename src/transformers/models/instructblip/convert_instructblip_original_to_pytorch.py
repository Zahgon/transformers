
import argparse
from io import BytesIO

import httpx
import torch

from lavis.models import load_model_and_preprocess
from PIL import Image

from transformers import (
    AutoTokenizer,
    BlipImageProcessor,
    InstructBlipConfig,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipQFormerConfig,
    InstructBlipVisionConfig,
    LlamaConfig,
    LlamaTokenizerFast,
    T5Config,
    T5TokenizerFast,
)
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


def load_demo_image():
    pass


def create_rename_keys(config):
    rename_keys = []

    rename_keys.append(("visual_encoder.cls_token", "vision_model.embeddings.class_embedding"))
    rename_keys.append(("visual_encoder.pos_embed", "vision_model.embeddings.position_embedding"))
    rename_keys.append(("visual_encoder.patch_embed.proj.weight", "vision_model.embeddings.patch_embedding.weight"))
    rename_keys.append(("visual_encoder.patch_embed.proj.bias", "vision_model.embeddings.patch_embedding.bias"))
    rename_keys.append(("ln_vision.weight", "vision_model.post_layernorm.weight"))
    rename_keys.append(("ln_vision.bias", "vision_model.post_layernorm.bias"))

    for i in range(config.vision_config.num_hidden_layers):
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.weight", f"vision_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.bias", f"vision_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.weight", f"vision_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.bias", f"vision_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.qkv.weight", f"vision_model.encoder.layers.{i}.self_attn.qkv.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.weight", f"vision_model.encoder.layers.{i}.self_attn.projection.weight",))
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.bias", f"vision_model.encoder.layers.{i}.self_attn.projection.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.weight", f"vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.bias", f"vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc2.weight", f"vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc2.bias", f"vision_model.encoder.layers.{i}.mlp.fc2.bias"))

    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.weight", "qformer.embeddings.layernorm.weight"))
    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.bias", "qformer.embeddings.layernorm.bias"))

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def read_in_q_v_bias(state_dict, config):
    pass


def get_blip2_config(model_name):
    pass


@torch.no_grad()
def convert_blip2_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = [
        "instructblip-vicuna-7b",
        "instructblip-vicuna-13b",
        "instructblip-flan-t5-xl",
        "instructblip-flan-t5-xxl",
    ]
    parser.add_argument(
        "--model_name",
        default="instructblip-flan-t5-xl",
        choices=choices,
        type=str,
        help="Path to hf config.json of model to convert",
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    args = parser.parse_args()

    convert_blip2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
