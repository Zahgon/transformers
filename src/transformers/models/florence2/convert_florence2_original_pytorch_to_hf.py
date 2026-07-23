import argparse
from collections import OrderedDict

import torch

from transformers import (
    AddedToken,
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    Florence2Config,
    Florence2ForConditionalGeneration,
    Florence2Processor,
    Florence2VisionConfig,
)


def convert_config(original_config: dict):
    new_config = Florence2VisionConfig(
        embed_dim=original_config["dim_embed"],
        max_temporal_embeddings=original_config["visual_temporal_embedding"]["max_temporal_embeddings"],
        max_pos_embeddings=original_config["image_pos_embed"]["max_pos_embeddings"],
        **original_config,
    )

    return new_config


def vision_conv_embeddings(idx):
    pass


def vision_spatial_block(stage_idx, block_idx):
    pass


def vision_channel_block(stage_idx, block_idx):
    pass


def multi_modal_projector():
    """
    Function helps in renaming final classification layer
    """
    projector = []
    projector.append(("image_projection", "model.multi_modal_projector.image_projection.weight"))
    projector.append(("image_proj_norm.weight", "model.multi_modal_projector.image_proj_norm.weight"))
    projector.append(("image_proj_norm.bias", "model.multi_modal_projector.image_proj_norm.bias"))
    projector.append(
        (
            "image_pos_embed.row_embeddings.weight",
            "model.multi_modal_projector.image_position_embed.row_embeddings.weight",
        )
    )
    projector.append(
        (
            "image_pos_embed.column_embeddings.weight",
            "model.multi_modal_projector.image_position_embed.column_embeddings.weight",
        )
    )
    projector.append(
        (
            "visual_temporal_embed.pos_idx_to_embed",
            "model.multi_modal_projector.visual_temporal_embed.pos_idx_to_embed",
        )
    )
    return projector


def language_model(state_dict):
    language_state_dict_keys = []
    for key in state_dict.keys():
        if key.startswith("language_model.model") and "lm_head" not in key:
            new_key = key.replace("language_model.model.", "model.language_model.")
            language_state_dict_keys.append((key, new_key))
    language_state_dict_keys.append(("language_model.lm_head.weight", "lm_head.weight"))
    return language_state_dict_keys


def convert_florence2_checkpoint(hf_model_id, pytorch_dump_folder, output_hub_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_id",
        default="microsoft/Florence-2-base",
        type=str,
        help="Name of the florence2 model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )

    args = parser.parse_args()
    convert_florence2_checkpoint(args.hf_model_id, args.pytorch_dump_folder_path, args.output_hub_path)
