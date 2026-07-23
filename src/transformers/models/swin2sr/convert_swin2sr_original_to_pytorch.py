
import argparse
from io import BytesIO

import httpx
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution, Swin2SRImageProcessor


def get_config(checkpoint_url):
    config = Swin2SRConfig()

    if "Swin2SR_ClassicalSR_X4_64" in checkpoint_url:
        config.upscale = 4
    elif "Swin2SR_CompressedSR_X4_48" in checkpoint_url:
        config.upscale = 4
        config.image_size = 48
        config.upsampler = "pixelshuffle_aux"
    elif "Swin2SR_Lightweight_X2_64" in checkpoint_url:
        config.depths = [6, 6, 6, 6]
        config.embed_dim = 60
        config.num_heads = [6, 6, 6, 6]
        config.upsampler = "pixelshuffledirect"
    elif "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR" in checkpoint_url:
        config.upscale = 4
        config.upsampler = "nearest+conv"
    elif "Swin2SR_Jpeg_dynamic" in checkpoint_url:
        config.num_channels = 1
        config.upscale = 1
        config.image_size = 126
        config.window_size = 7
        config.img_range = 255.0
        config.upsampler = ""

    return config


def rename_key(name, config):
    if "patch_embed.proj" in name and "layers" not in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.patch_embeddings.layernorm")
    if "layers" in name:
        name = name.replace("layers", "encoder.stages")
    if "residual_group.blocks" in name:
        name = name.replace("residual_group.blocks", "layers")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "q_bias" in name:
        name = name.replace("q_bias", "query.bias")
    if "k_bias" in name:
        name = name.replace("k_bias", "key.bias")
    if "v_bias" in name:
        name = name.replace("v_bias", "value.bias")
    if "cpb_mlp" in name:
        name = name.replace("cpb_mlp", "continuous_position_bias_mlp")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "patch_embed.projection")

    if name == "norm.weight":
        name = "layernorm.weight"
    if name == "norm.bias":
        name = "layernorm.bias"

    if "conv_first" in name:
        name = name.replace("conv_first", "first_convolution")

    if (
        "upsample" in name
        or "conv_before_upsample" in name
        or "conv_bicubic" in name
        or "conv_up" in name
        or "conv_hr" in name
        or "conv_last" in name
        or "aux" in name
    ):
        if "conv_last" in name:
            name = name.replace("conv_last", "final_convolution")
        if config.upsampler in ["pixelshuffle", "pixelshuffle_aux", "nearest+conv"]:
            if "conv_before_upsample.0" in name:
                name = name.replace("conv_before_upsample.0", "conv_before_upsample")
            if "upsample.0" in name:
                name = name.replace("upsample.0", "upsample.convolution_0")
            if "upsample.2" in name:
                name = name.replace("upsample.2", "upsample.convolution_1")
            name = "upsample." + name
        elif config.upsampler == "pixelshuffledirect":
            name = name.replace("upsample.0.weight", "upsample.conv.weight")
            name = name.replace("upsample.0.bias", "upsample.conv.bias")
        else:
            pass
    else:
        name = "swin2sr." + name

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            stage_num = int(key_split[1])
            block_num = int(key_split[4])
            dim = config.embed_dim

            if "weight" in key:
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.key.weight"] = (
                    val[dim : dim * 2, :]
                )
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.query.bias"] = (
                    val[:dim]
                )
                orig_state_dict[f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.key.bias"] = (
                    val[dim : dim * 2]
                )
                orig_state_dict[f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.value.bias"] = (
                    val[-dim:]
                )
        else:
            orig_state_dict[rename_key(key, config)] = val

    return orig_state_dict


def convert_swin2sr_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth",
        type=str,
        help="URL of the original Swin2SR checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the converted model to the hub.")

    args = parser.parse_args()
    convert_swin2sr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
