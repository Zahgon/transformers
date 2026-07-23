
import argparse
import os

import torch

from transformers import FlavaConfig, FlavaForPreTraining
from transformers.models.flava.convert_dalle_to_flava_codebook import convert_dalle_checkpoint


def count_parameters(state_dict):
    return sum(param.float().sum() if "encoder.embeddings" not in key else 0 for key, param in state_dict.items())


def upgrade_state_dict(state_dict, codebook_state_dict):
    upgrade = {}

    for key, value in state_dict.items():
        if "text_encoder.embeddings" in key or "image_encoder.embeddings" in key:
            continue

        key = key.replace("heads.cmd.mim_head.cls.predictions", "mmm_image_head")
        key = key.replace("heads.cmd.mlm_head.cls.predictions", "mmm_text_head")
        key = key.replace("heads.cmd.itm_head.cls", "itm_head")
        key = key.replace("heads.cmd.itm_head.pooler", "itm_head.pooler")
        key = key.replace("heads.cmd.clip_head.logit_scale", "flava.logit_scale")
        key = key.replace("heads.fairseq_mlm.cls.predictions", "mlm_head")
        key = key.replace("heads.imagenet.mim_head.cls.predictions", "mim_head")
        key = key.replace("mm_text_projection", "flava.text_to_mm_projection")
        key = key.replace("mm_image_projection", "flava.image_to_mm_projection")
        key = key.replace("image_encoder.module", "flava.image_model")
        key = key.replace("text_encoder.module", "flava.text_model")
        key = key.replace("mm_encoder.module.encoder.cls_token", "flava.multimodal_model.cls_token")
        key = key.replace("mm_encoder.module", "flava.multimodal_model")
        key = key.replace("text_projection", "flava.text_projection")
        key = key.replace("image_projection", "flava.image_projection")

        upgrade[key] = value.float()

    for key, value in codebook_state_dict.items():
        upgrade[f"image_codebook.{key}"] = value

    return upgrade


@torch.no_grad()
def convert_flava_checkpoint(checkpoint_path, codebook_path, pytorch_dump_folder_path, config_path=None):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to flava checkpoint")
    parser.add_argument("--codebook_path", default=None, type=str, help="Path to flava codebook checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    convert_flava_checkpoint(args.checkpoint_path, args.codebook_path, args.pytorch_dump_folder_path, args.config_path)
