
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import DPTConfig, DPTForDepthEstimation, DPTForSemanticSegmentation, DPTImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(checkpoint_url):
    pass


def remove_ignore_keys_(state_dict):
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    if (
        "pretrained.model" in name
        and "cls_token" not in name
        and "pos_embed" not in name
        and "patch_embed" not in name
    ):
        name = name.replace("pretrained.model", "dpt.encoder")
    if "pretrained.model" in name:
        name = name.replace("pretrained.model", "dpt.embeddings")
    if "patch_embed" in name:
        name = name.replace("patch_embed", "")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "position_embeddings")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "proj" in name and "project" not in name:
        name = name.replace("proj", "projection")
    if "blocks" in name:
        name = name.replace("blocks", "layer")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "norm1" in name and "backbone" not in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name and "backbone" not in name:
        name = name.replace("norm2", "layernorm_after")
    if "scratch.output_conv" in name:
        name = name.replace("scratch.output_conv", "head")
    if "scratch" in name:
        name = name.replace("scratch", "neck")
    if "layer1_rn" in name:
        name = name.replace("layer1_rn", "convs.0")
    if "layer2_rn" in name:
        name = name.replace("layer2_rn", "convs.1")
    if "layer3_rn" in name:
        name = name.replace("layer3_rn", "convs.2")
    if "layer4_rn" in name:
        name = name.replace("layer4_rn", "convs.3")
    if "refinenet" in name:
        layer_idx = int(name[len("neck.refinenet") : len("neck.refinenet") + 1])
        name = name.replace(f"refinenet{layer_idx}", f"fusion_stage.layers.{abs(layer_idx - 4)}")
    if "out_conv" in name:
        name = name.replace("out_conv", "projection")
    if "resConfUnit1" in name:
        name = name.replace("resConfUnit1", "residual_layer1")
    if "resConfUnit2" in name:
        name = name.replace("resConfUnit2", "residual_layer2")
    if "conv1" in name:
        name = name.replace("conv1", "convolution1")
    if "conv2" in name:
        name = name.replace("conv2", "convolution2")
    if "pretrained.act_postprocess1.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess1.0.project.0", "neck.reassemble_stage.readout_projects.0.0")
    if "pretrained.act_postprocess2.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess2.0.project.0", "neck.reassemble_stage.readout_projects.1.0")
    if "pretrained.act_postprocess3.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess3.0.project.0", "neck.reassemble_stage.readout_projects.2.0")
    if "pretrained.act_postprocess4.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess4.0.project.0", "neck.reassemble_stage.readout_projects.3.0")

    if "pretrained.act_postprocess1.3" in name:
        name = name.replace("pretrained.act_postprocess1.3", "neck.reassemble_stage.layers.0.projection")
    if "pretrained.act_postprocess1.4" in name:
        name = name.replace("pretrained.act_postprocess1.4", "neck.reassemble_stage.layers.0.resize")
    if "pretrained.act_postprocess2.3" in name:
        name = name.replace("pretrained.act_postprocess2.3", "neck.reassemble_stage.layers.1.projection")
    if "pretrained.act_postprocess2.4" in name:
        name = name.replace("pretrained.act_postprocess2.4", "neck.reassemble_stage.layers.1.resize")
    if "pretrained.act_postprocess3.3" in name:
        name = name.replace("pretrained.act_postprocess3.3", "neck.reassemble_stage.layers.2.projection")
    if "pretrained.act_postprocess4.3" in name:
        name = name.replace("pretrained.act_postprocess4.3", "neck.reassemble_stage.layers.3.projection")
    if "pretrained.act_postprocess4.4" in name:
        name = name.replace("pretrained.act_postprocess4.4", "neck.reassemble_stage.layers.3.resize")
    if "pretrained" in name:
        name = name.replace("pretrained", "dpt")
    if "bn" in name:
        name = name.replace("bn", "batch_norm")
    if "head" in name:
        name = name.replace("head", "head.head")
    if "encoder.norm" in name:
        name = name.replace("encoder.norm", "layernorm")
    if "auxlayer" in name:
        name = name.replace("auxlayer", "auxiliary_head.head")
    if "backbone" in name:
        name = name.replace("backbone", "backbone.bit.encoder")

    if ".." in name:
        name = name.replace("..", ".")

    if "stem.conv" in name:
        name = name.replace("stem.conv", "bit.embedder.convolution")
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    if "convolution" in name and "backbone" in name:
        name = name.replace("convolution", "conv")
    if "layer" in name and "backbone" in name:
        name = name.replace("layer", "layers")
    if "backbone.bit.encoder.bit" in name:
        name = name.replace("backbone.bit.encoder.bit", "backbone.bit")
    if "embedder.conv" in name:
        name = name.replace("embedder.conv", "embedder.convolution")
    if "backbone.bit.encoder.stem.norm" in name:
        name = name.replace("backbone.bit.encoder.stem.norm", "backbone.bit.embedder.norm")
    return name


def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        in_proj_weight = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.bias")
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_dpt_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub, model_name, show_prediction):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        type=str,
        help="URL of the original DPT checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    parser.add_argument(
        "--model_name",
        default="dpt-large",
        type=str,
        help="Name of the model, in case you're pushing to the hub.",
    )
    parser.add_argument(
        "--show_prediction",
        action="store_true",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub, args.model_name, args.show_prediction
    )
