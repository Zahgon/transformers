
import argparse
import os
import re
from io import BytesIO

import httpx
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import DINOv3ViTConfig, DINOv3ViTImageProcessorFast, DINOv3ViTModel


HUB_MODELS = {
    "vits16_lvd1689m": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus_lvd1689m": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16_lvd1689m": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16_lvd1689m": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vitl16_sat493m": "facebook/dinov3-vitl16-pretrain-sat493m",
    "vith16plus_lvd1689m": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vit7b16_lvd1689m": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "vit7b16_sat493m": "facebook/dinov3-vit7b16-pretrain-sat493m",
    "eupe_vitt16": "facebook/EUPE-ViT-T",
    "eupe_vits16": "facebook/EUPE-ViT-S",
    "eupe_vitb16": "facebook/EUPE-ViT-B",
}

HUB_CHECKPOINTS = {
    "vits16_lvd1689m": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "vits16plus_lvd1689m": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "vitb16_lvd1689m": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "vitl16_lvd1689m": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "vitl16_sat493m": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    "vith16plus_lvd1689m": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "vit7b16_lvd1689m": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "vit7b16_sat493m": "dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
    "eupe_vitt16": "EUPE-ViT-T.pt",
    "eupe_vits16": "EUPE-ViT-S.pt",
    "eupe_vitb16": "EUPE-ViT-B.pt",
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"cls_token":                   r"embeddings.cls_token",
    r"mask_token":                  r"embeddings.mask_token",
    r"storage_tokens":              r"embeddings.register_tokens",
    r"patch_embed.proj":            r"embeddings.patch_embeddings",
    r"periods":                     r"inv_freq",
    r"rope_embed":                  r"rope_embeddings",
    r"blocks.(\d+).attn.proj":      r"layer.\1.attention.o_proj",
    r"blocks.(\d+).attn.":          r"layer.\1.attention.",
    r"blocks.(\d+).ls(\d+).gamma":  r"layer.\1.layer_scale\2.lambda1",
    r"blocks.(\d+).mlp.fc1":        r"layer.\1.mlp.up_proj",
    r"blocks.(\d+).mlp.fc2":        r"layer.\1.mlp.down_proj",
    r"blocks.(\d+).mlp":            r"layer.\1.mlp",
    r"blocks.(\d+).norm":           r"layer.\1.norm",
    r"w1":                          r"gate_proj",
    r"w2":                          r"up_proj",
    r"w3":                          r"down_proj",
}


def convert_old_keys_to_new_keys(state_dict_keys: dict | None = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def split_qkv(state_dict: dict):
    keys = [x for x in state_dict.keys() if "qkv" in x]
    for key in keys:
        qkv = state_dict.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        state_dict[key.replace("qkv", "q_proj")] = q
        state_dict[key.replace("qkv", "k_proj")] = k
        state_dict[key.replace("qkv", "v_proj")] = v
    return state_dict


def get_dinov3_config(model_name: str) -> DINOv3ViTConfig:
    if model_name == "vits16_lvd1689m":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=12,
            num_attention_heads=6,
            proj_bias=True,
            num_register_tokens=4,
            use_gated_mlp=False,
            hidden_act="gelu",
        )
    elif model_name == "vits16plus_lvd1689m":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=12,
            num_attention_heads=6,
            num_register_tokens=4,
            use_gated_mlp=True,
            hidden_act="silu",
        )
    elif model_name == "vitb16_lvd1689m":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            proj_bias=True,
            num_register_tokens=4,
            use_gated_mlp=False,
            hidden_act="gelu",
        )
    elif model_name in ("vitl16_lvd1689m", "vitl16_sat493m"):
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_register_tokens=4,
            use_gated_mlp=False,
            hidden_act="gelu",
        )
    elif model_name == "vith16plus_lvd1689m":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=1280,
            intermediate_size=5120,
            num_hidden_layers=32,
            num_attention_heads=20,
            num_register_tokens=4,
            use_gated_mlp=True,
            hidden_act="silu",
        )
    elif model_name in ("vit7b16_lvd1689m", "vit7b16_sat493m"):
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=4096,
            intermediate_size=8192,
            num_hidden_layers=40,
            num_attention_heads=32,
            query_bias=False,
            value_bias=False,
            num_register_tokens=4,
            use_gated_mlp=True,
            hidden_act="silu",
        )
    elif model_name in ("eupe_vitt16", "eupe_vits16", "eupe_vitb16"):
        hidden_size, num_attention_heads = {
            "eupe_vitt16": (192, 3),
            "eupe_vits16": (384, 6),
            "eupe_vitb16": (768, 12),
        }[model_name]
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=12,
            num_attention_heads=num_attention_heads,
            num_register_tokens=4,
            use_gated_mlp=False,
            hidden_act="gelu",
            layerscale_value=1e-5,
        )
    else:
        raise ValueError("Model not supported")


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read())).convert("RGB")
    return image


def get_transform(resize_size: int = 224):
    pass


def get_image_processor(resize_size: int = 224):
    pass


@torch.no_grad()
def convert_and_test_dinov3_checkpoint(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="vith16plus_lvd1689m",
        type=str,
        choices=[
            "vits16_lvd1689m",
            "vits16plus_lvd1689m",
            "vitb16_lvd1689m",
            "vitl16_lvd1689m",
            "vitl16_sat493m",
            "vith16plus_lvd1689m",
            "vit7b16_lvd1689m",
            "vit7b16_sat493m",
            "eupe_vitt16",
            "eupe_vits16",
            "eupe_vitb16",
        ],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--save-dir",
        default="converted_models",
        type=str,
        help="Directory to save the converted model.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the converted model to the Hugging Face Hub.",
    )
    args = parser.parse_args()
    convert_and_test_dinov3_checkpoint(args)
