
import argparse
import collections
import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.training import checkpoints
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    CLIPTokenizer,
    Owlv2Config,
    Owlv2ForObjectDetection,
    Owlv2ImageProcessor,
    Owlv2Processor,
    Owlv2TextConfig,
    Owlv2VisionConfig,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_owlv2_config(model_name):
    pass


def flatten_nested_dict(params, parent_key="", sep="/"):
    pass


def create_rename_keys(config, model_name):
    rename_keys = []

    rename_keys.append(("backbone/clip/visual/class_embedding", "owlv2.vision_model.embeddings.class_embedding"))
    rename_keys.append(("backbone/clip/visual/conv1/kernel", "owlv2.vision_model.embeddings.patch_embedding.weight"))
    rename_keys.append(("backbone/clip/visual/positional_embedding", "owlv2.vision_model.embeddings.position_embedding.weight"))
    rename_keys.append(("backbone/clip/visual/ln_pre/scale", "owlv2.vision_model.pre_layernorm.weight"))
    rename_keys.append(("backbone/clip/visual/ln_pre/bias", "owlv2.vision_model.pre_layernorm.bias"))

    for i in range(config.vision_config.num_hidden_layers):
        if "v2" in model_name:
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_0/scale", f"owlv2.vision_model.encoder.layers.{i}.layer_norm1.weight"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_0/bias", f"owlv2.vision_model.encoder.layers.{i}.layer_norm1.bias"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_1/scale", f"owlv2.vision_model.encoder.layers.{i}.layer_norm2.weight"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_1/bias", f"owlv2.vision_model.encoder.layers.{i}.layer_norm2.bias"))
        else:
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_1/scale", f"owlv2.vision_model.encoder.layers.{i}.layer_norm1.weight"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_1/bias", f"owlv2.vision_model.encoder.layers.{i}.layer_norm1.bias"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_2/scale", f"owlv2.vision_model.encoder.layers.{i}.layer_norm2.weight"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_2/bias", f"owlv2.vision_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/mlp/c_fc/kernel", f"owlv2.vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/mlp/c_fc/bias", f"owlv2.vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/mlp/c_proj/kernel", f"owlv2.vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/mlp/c_proj/bias", f"owlv2.vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/query/kernel", f"owlv2.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/query/bias", f"owlv2.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/key/kernel", f"owlv2.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/key/bias", f"owlv2.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/value/kernel", f"owlv2.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/value/bias", f"owlv2.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/out/kernel", f"owlv2.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/out/bias", f"owlv2.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))

    rename_keys.append(("backbone/clip/visual/ln_post/scale", "owlv2.vision_model.post_layernorm.weight"))
    rename_keys.append(("backbone/clip/visual/ln_post/bias", "owlv2.vision_model.post_layernorm.bias"))

    rename_keys.append(("backbone/clip/text/token_embedding/embedding", "owlv2.text_model.embeddings.token_embedding.weight"))
    rename_keys.append(("backbone/clip/text/positional_embedding", "owlv2.text_model.embeddings.position_embedding.weight"))

    for i in range(config.text_config.num_hidden_layers):
        if "v2" in model_name:
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_0/scale", f"owlv2.text_model.encoder.layers.{i}.layer_norm1.weight"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_0/bias", f"owlv2.text_model.encoder.layers.{i}.layer_norm1.bias"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_1/scale", f"owlv2.text_model.encoder.layers.{i}.layer_norm2.weight"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_1/bias", f"owlv2.text_model.encoder.layers.{i}.layer_norm2.bias"))
        else:
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_1/scale", f"owlv2.text_model.encoder.layers.{i}.layer_norm1.weight"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_1/bias", f"owlv2.text_model.encoder.layers.{i}.layer_norm1.bias"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_2/scale", f"owlv2.text_model.encoder.layers.{i}.layer_norm2.weight"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_2/bias", f"owlv2.text_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/mlp/c_fc/kernel", f"owlv2.text_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/mlp/c_fc/bias", f"owlv2.text_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/mlp/c_proj/kernel", f"owlv2.text_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/mlp/c_proj/bias", f"owlv2.text_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/query/kernel", f"owlv2.text_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/query/bias", f"owlv2.text_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/key/kernel", f"owlv2.text_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/key/bias", f"owlv2.text_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/value/kernel", f"owlv2.text_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/value/bias", f"owlv2.text_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/out/kernel", f"owlv2.text_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/out/bias", f"owlv2.text_model.encoder.layers.{i}.self_attn.out_proj.bias"))

    rename_keys.append(("backbone/clip/text/ln_final/scale", "owlv2.text_model.final_layer_norm.weight"))
    rename_keys.append(("backbone/clip/text/ln_final/bias", "owlv2.text_model.final_layer_norm.bias"))

    rename_keys.append(("backbone/clip/logit_scale", "owlv2.logit_scale"))

    rename_keys.append(("backbone/clip/text/text_projection/kernel", "owlv2.text_projection.weight"))

    rename_keys.append(("backbone/merged_class_token/scale", "layer_norm.weight"))
    rename_keys.append(("backbone/merged_class_token/bias", "layer_norm.bias"))
    rename_keys.append(("class_head/Dense_0/kernel", "class_head.dense0.weight"))
    rename_keys.append(("class_head/Dense_0/bias", "class_head.dense0.bias"))
    rename_keys.append(("class_head/logit_shift/kernel", "class_head.logit_shift.weight"))
    rename_keys.append(("class_head/logit_scale/kernel", "class_head.logit_scale.weight"))
    rename_keys.append(("class_head/logit_scale/bias", "class_head.logit_scale.bias"))
    rename_keys.append(("class_head/logit_shift/bias", "class_head.logit_shift.bias"))
    rename_keys.append(("obj_box_head/Dense_0/kernel", "box_head.dense0.weight"))
    rename_keys.append(("obj_box_head/Dense_0/bias", "box_head.dense0.bias"))
    rename_keys.append(("obj_box_head/Dense_1/kernel", "box_head.dense1.weight"))
    rename_keys.append(("obj_box_head/Dense_1/bias", "box_head.dense1.bias"))
    rename_keys.append(("obj_box_head/Dense_2/kernel", "box_head.dense2.weight"))
    rename_keys.append(("obj_box_head/Dense_2/bias", "box_head.dense2.bias"))

    if "v2" in model_name:
        rename_keys.append(("objectness_head/Dense_0/kernel", "objectness_head.dense0.weight"))
        rename_keys.append(("objectness_head/Dense_0/bias", "objectness_head.dense0.bias"))
        rename_keys.append(("objectness_head/Dense_1/kernel", "objectness_head.dense1.weight"))
        rename_keys.append(("objectness_head/Dense_1/bias", "objectness_head.dense1.bias"))
        rename_keys.append(("objectness_head/Dense_2/kernel", "objectness_head.dense2.weight"))
        rename_keys.append(("objectness_head/Dense_2/bias", "objectness_head.dense2.bias"))


    return rename_keys


def rename_and_reshape_key(dct, old, new, config):
    pass


@torch.no_grad()
def convert_owlv2_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub, verify_logits):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="owlv2-base-patch16",
        choices=[
            "owlv2-base-patch16",
            "owlv2-base-patch16-finetuned",
            "owlv2-base-patch16-ensemble",
            "owlv2-large-patch14",
            "owlv2-large-patch14-finetuned",
            "owlv2-large-patch14-ensemble",
        ],
        type=str,
        help="Name of the Owlv2 model you'd like to convert from FLAX to PyTorch.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the original Flax checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--verify_logits",
        action="store_false",
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image preprocessor to the hub")

    args = parser.parse_args()
    convert_owlv2_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits
    )
