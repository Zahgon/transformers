
import argparse
import re
from io import BytesIO

import httpx
import torch

from models.blip import blip_decoder
from models.blip_itm import blip_itm
from models.blip_vqa import blip_vqa
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from transformers import (
    BertTokenizer,
    BlipConfig,
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval,
    BlipForQuestionAnswering,
)


def load_demo_image(image_size, device):
    pass


def rename_key(key):
    if "visual_encoder" in key:
        key = re.sub("visual_encoder*", "vision_model.encoder", key)
    if "blocks" in key:
        key = re.sub(r"blocks", "layers", key)
    if "attn" in key:
        key = re.sub(r"attn", "self_attn", key)
    if "norm1" in key:
        key = re.sub(r"norm1", "layer_norm1", key)
    if "norm2" in key:
        key = re.sub(r"norm2", "layer_norm2", key)
    if "encoder.norm" in key:
        key = re.sub(r"encoder.norm", "post_layernorm", key)
    if "encoder.patch_embed.proj" in key:
        key = re.sub(r"encoder.patch_embed.proj", "embeddings.patch_embedding", key)

    if "encoder.pos_embed" in key:
        key = re.sub(r"encoder.pos_embed", "embeddings.position_embedding", key)
    if "encoder.cls_token" in key:
        key = re.sub(r"encoder.cls_token", "embeddings.class_embedding", key)

    if "self_attn" in key:
        key = re.sub(r"self_attn.proj", "self_attn.projection", key)

    return key


@torch.no_grad()
def convert_blip_checkpoint(pytorch_dump_folder_path, config_path=None):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    convert_blip_checkpoint(args.pytorch_dump_folder_path, args.config_path)
