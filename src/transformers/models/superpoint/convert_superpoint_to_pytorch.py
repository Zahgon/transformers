import argparse
import os
from io import BytesIO

import httpx
import torch
from PIL import Image

from transformers import SuperPointConfig, SuperPointForKeypointDetection, SuperPointImageProcessor


def get_superpoint_config():
    pass


def create_rename_keys(config, state_dict):
    rename_keys = []

    rename_keys.append(("conv1a.weight", "encoder.conv_blocks.0.conv_a.weight"))
    rename_keys.append(("conv1b.weight", "encoder.conv_blocks.0.conv_b.weight"))
    rename_keys.append(("conv2a.weight", "encoder.conv_blocks.1.conv_a.weight"))
    rename_keys.append(("conv2b.weight", "encoder.conv_blocks.1.conv_b.weight"))
    rename_keys.append(("conv3a.weight", "encoder.conv_blocks.2.conv_a.weight"))
    rename_keys.append(("conv3b.weight", "encoder.conv_blocks.2.conv_b.weight"))
    rename_keys.append(("conv4a.weight", "encoder.conv_blocks.3.conv_a.weight"))
    rename_keys.append(("conv4b.weight", "encoder.conv_blocks.3.conv_b.weight"))
    rename_keys.append(("conv1a.bias", "encoder.conv_blocks.0.conv_a.bias"))
    rename_keys.append(("conv1b.bias", "encoder.conv_blocks.0.conv_b.bias"))
    rename_keys.append(("conv2a.bias", "encoder.conv_blocks.1.conv_a.bias"))
    rename_keys.append(("conv2b.bias", "encoder.conv_blocks.1.conv_b.bias"))
    rename_keys.append(("conv3a.bias", "encoder.conv_blocks.2.conv_a.bias"))
    rename_keys.append(("conv3b.bias", "encoder.conv_blocks.2.conv_b.bias"))
    rename_keys.append(("conv4a.bias", "encoder.conv_blocks.3.conv_a.bias"))
    rename_keys.append(("conv4b.bias", "encoder.conv_blocks.3.conv_b.bias"))

    rename_keys.append(("convPa.weight", "keypoint_decoder.conv_score_a.weight"))
    rename_keys.append(("convPb.weight", "keypoint_decoder.conv_score_b.weight"))
    rename_keys.append(("convPa.bias", "keypoint_decoder.conv_score_a.bias"))
    rename_keys.append(("convPb.bias", "keypoint_decoder.conv_score_b.bias"))

    rename_keys.append(("convDa.weight", "descriptor_decoder.conv_descriptor_a.weight"))
    rename_keys.append(("convDb.weight", "descriptor_decoder.conv_descriptor_b.weight"))
    rename_keys.append(("convDa.bias", "descriptor_decoder.conv_descriptor_a.bias"))
    rename_keys.append(("convDb.bias", "descriptor_decoder.conv_descriptor_b.bias"))

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def prepare_imgs():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image_1 = Image.open(BytesIO(response.read()))
    url = "http://images.cocodataset.org/test-stuff2017/000000004016.jpg"
    with httpx.stream("GET", url) as response:
        image_2 = Image.open(BytesIO(response.read()))
    return [image_1, image_2]


@torch.no_grad()
def convert_superpoint_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub, test_mode=False):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth",
        type=str,
        help="URL of the original SuperPoint checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image preprocessor to the hub")

    args = parser.parse_args()
    convert_superpoint_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub
    )
