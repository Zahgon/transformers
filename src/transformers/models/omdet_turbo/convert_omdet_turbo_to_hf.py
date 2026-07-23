
import argparse
from io import BytesIO

import httpx
import torch
from PIL import Image

from transformers import (
    CLIPTokenizer,
    DetrImageProcessor,
    OmDetTurboConfig,
    OmDetTurboForObjectDetection,
    OmDetTurboProcessor,
)


IMAGE_MEAN = [123.675, 116.28, 103.53]
IMAGE_STD = [58.395, 57.12, 57.375]


def get_omdet_turbo_config(model_name, use_timm_backbone):
    pass


def create_rename_keys_vision(state_dict, config):
    pass


def create_rename_keys_language(state_dict):
    pass


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def read_in_q_k_v_vision(state_dict, config):
    pass


def read_in_q_k_v_text(state_dict, config):
    pass


def read_in_q_k_v_encoder(state_dict, config):
    pass


def read_in_q_k_v_decoder(state_dict, config):
    pass


def run_test(model, processor):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read())).convert("RGB")

    classes = ["cat", "remote"]
    task = f"Detect {', '.join(classes)}."
    inputs = processor(image, text=classes, task=task, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_slice = outputs[1][0, :3, :3]
    print(predicted_slice)
    expected_slice = torch.tensor([[0.9427, -2.5958], [0.2105, -3.4569], [-2.6364, -4.1610]])

    assert torch.allclose(predicted_slice, expected_slice, atol=1e-4)
    print("Looks ok!")


@torch.no_grad()
def convert_omdet_turbo_checkpoint(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="omdet-turbo-tiny",
        type=str,
        choices=["omdet-turbo-tiny"],
        help="Name of the OmDetTurbo model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )
    parser.add_argument(
        "--use_timm_backbone", action="store_true", help="Whether or not to use timm backbone for vision backbone."
    )

    args = parser.parse_args()
    convert_omdet_turbo_checkpoint(args)
