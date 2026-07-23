
import argparse
import os
import re
from io import BytesIO

import httpx
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel, DINOv3ViTImageProcessorFast


HUB_MODELS = {
    "convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "convnext_base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    "eupe_convnext_tiny": "facebook/EUPE-ConvNeXt-T",
    "eupe_convnext_small": "facebook/EUPE-ConvNeXt-S",
    "eupe_convnext_base": "facebook/EUPE-ConvNeXt-B",
}

HUB_CHECKPOINTS = {
    "convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "convnext_small": "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "convnext_base": "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "convnext_large": "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
    "eupe_convnext_tiny": "EUPE-ConvNeXt-T.pt",
    "eupe_convnext_small": "EUPE-ConvNeXt-S.pt",
    "eupe_convnext_base": "EUPE-ConvNeXt-B.pt",
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"dwconv":                              r"depthwise_conv",
    r"pwconv":                              r"pointwise_conv",
    r"norm":                                r"layer_norm",
    r"stages.(\d+).(\d+)":                  r"stages.\1.layers.\2",
    r"downsample_layers.(\d+).(\d+)":       r"stages.\1.downsample_layers.\2",
}


def get_dinov3_config(model_name: str) -> DINOv3ConvNextConfig:
    model_name = model_name.removeprefix("eupe_")
    if model_name == "convnext_tiny":
        return DINOv3ConvNextConfig(
            depths=[3, 3, 9, 3],
            hidden_sizes=[96, 192, 384, 768],
        )
    elif model_name == "convnext_small":
        return DINOv3ConvNextConfig(
            depths=[3, 3, 27, 3],
            hidden_sizes=[96, 192, 384, 768],
        )
    elif model_name == "convnext_base":
        return DINOv3ConvNextConfig(
            depths=[3, 3, 27, 3],
            hidden_sizes=[128, 256, 512, 1024],
        )
    elif model_name == "convnext_large":
        return DINOv3ConvNextConfig(
            depths=[3, 3, 27, 3],
            hidden_sizes=[192, 384, 768, 1536],
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


@torch.no_grad()
def convert_and_test_dinov3_checkpoint(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="convnext_tiny",
        type=str,
        choices=[
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
            "eupe_convnext_tiny",
            "eupe_convnext_small",
            "eupe_convnext_base",
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
