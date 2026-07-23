

import argparse
from io import BytesIO

import httpx
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import SamHQConfig, SamHQModel, SamHQProcessor, SamHQVisionConfig, SamImageProcessor


def get_config(model_name):
    if "sam_hq_vit_b" in model_name:
        vision_config = SamHQVisionConfig()
        vit_dim = 768  # Base model dimension
    elif "sam_hq_vit_l" in model_name:
        vision_config = SamHQVisionConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            global_attn_indexes=[5, 11, 17, 23],
        )
        vit_dim = 1024  # Large model dimension
    elif "sam_hq_vit_h" in model_name:
        vision_config = SamHQVisionConfig(
            hidden_size=1280,
            num_hidden_layers=32,
            num_attention_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
        )
        vit_dim = 1280  # Huge model dimension

    mask_decoder_config = {"vit_dim": vit_dim}

    config = SamHQConfig(
        vision_config=vision_config,
        mask_decoder_config=mask_decoder_config,
    )

    return config


KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "image_encoder": "vision_encoder",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "patch_embed.proj": "patch_embed.projection",
    ".norm": ".layer_norm",
    "blocks": "layers",
    "mask_decoder.hf_token": "mask_decoder.hq_token",
    "mask_decoder.compress_vit_feat.0": "mask_decoder.compress_vit_conv1",
    "mask_decoder.compress_vit_feat.1": "mask_decoder.compress_vit_norm",
    "mask_decoder.compress_vit_feat.3": "mask_decoder.compress_vit_conv2",
    "mask_decoder.embedding_encoder.0": "mask_decoder.encoder_conv1",
    "mask_decoder.embedding_encoder.1": "mask_decoder.encoder_norm",
    "mask_decoder.embedding_encoder.3": "mask_decoder.encoder_conv2",
    "mask_decoder.embedding_maskfeature.0": "mask_decoder.mask_conv1",
    "mask_decoder.embedding_maskfeature.1": "mask_decoder.mask_norm",
    "mask_decoder.embedding_maskfeature.3": "mask_decoder.mask_conv2",
    "mask_decoder.hf_mlp": "mask_decoder.hq_mask_mlp",
    "output_hypernetworks_mlps.0.layers.0": "output_hypernetworks_mlps.0.proj_in",
    "output_hypernetworks_mlps.0.layers.1": "output_hypernetworks_mlps.0.layers.0",
    "output_hypernetworks_mlps.0.layers.2": "output_hypernetworks_mlps.0.proj_out",
    "output_hypernetworks_mlps.1.layers.0": "output_hypernetworks_mlps.1.proj_in",
    "output_hypernetworks_mlps.1.layers.1": "output_hypernetworks_mlps.1.layers.0",
    "output_hypernetworks_mlps.1.layers.2": "output_hypernetworks_mlps.1.proj_out",
    "output_hypernetworks_mlps.2.layers.0": "output_hypernetworks_mlps.2.proj_in",
    "output_hypernetworks_mlps.2.layers.1": "output_hypernetworks_mlps.2.layers.0",
    "output_hypernetworks_mlps.2.layers.2": "output_hypernetworks_mlps.2.proj_out",
    "output_hypernetworks_mlps.3.layers.0": "output_hypernetworks_mlps.3.proj_in",
    "output_hypernetworks_mlps.3.layers.1": "output_hypernetworks_mlps.3.layers.0",
    "output_hypernetworks_mlps.3.layers.2": "output_hypernetworks_mlps.3.proj_out",
    "hq_mask_mlp.layers.0": "hq_mask_mlp.proj_in",
    "hq_mask_mlp.layers.1": "hq_mask_mlp.layers.0",
    "hq_mask_mlp.layers.2": "hq_mask_mlp.proj_out",
}


def replace_keys(state_dict):
    model_state_dict = {}
    state_dict.pop("pixel_mean", None)
    state_dict.pop("pixel_std", None)

    for key, value in state_dict.items():
        new_key = key

        for key_to_modify, replacement in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in new_key:
                new_key = new_key.replace(key_to_modify, replacement)

        model_state_dict[new_key] = value

    if "prompt_encoder.shared_embedding.positional_embedding" in model_state_dict:
        model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
            "prompt_encoder.shared_embedding.positional_embedding"
        ]

    if (
        "mask_decoder.iou_prediction_head.layers.0.weight" not in model_state_dict
        and "mask_decoder.iou_prediction_head.proj_in.weight" in model_state_dict
    ):
        model_state_dict["mask_decoder.iou_prediction_head.layers.0.weight"] = model_state_dict[
            "mask_decoder.iou_prediction_head.proj_in.weight"
        ]
        model_state_dict["mask_decoder.iou_prediction_head.layers.0.bias"] = model_state_dict[
            "mask_decoder.iou_prediction_head.proj_in.bias"
        ]

    return model_state_dict


def convert_sam_hq_checkpoint(model_name, checkpoint_path, pytorch_dump_folder, push_to_hub, hub_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = ["sam_hq_vit_b", "sam_hq_vit_h", "sam_hq_vit_l"]
    parser.add_argument(
        "--model_name",
        choices=choices,
        type=str,
        required=True,
        help="Name of the SAM-HQ model to convert",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the SAM-HQ checkpoint (.pth file)",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        default=None,
        help="Path to save the converted model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the converted model to the hub",
    )
    parser.add_argument(
        "--hub_path",
        type=str,
        default="sushmanth",
        help="Hugging Face Hub path where the model will be uploaded",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download("lkeab/hq-sam", f"{args.model_name}.pth")

    convert_sam_hq_checkpoint(
        args.model_name,
        checkpoint_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
        args.hub_path,
    )
