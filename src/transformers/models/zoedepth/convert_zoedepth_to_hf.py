
import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import BeitConfig, ZoeDepthConfig, ZoeDepthForDepthEstimation, ZoeDepthImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_zoedepth_config(model_name):
    pass


def rename_key(name):
    if "core.core.pretrained.model.blocks" in name:
        name = name.replace("core.core.pretrained.model.blocks", "backbone.encoder.layer")
    if "core.core.pretrained.model.patch_embed.proj" in name:
        name = name.replace(
            "core.core.pretrained.model.patch_embed.proj", "backbone.embeddings.patch_embeddings.projection"
        )
    if "core.core.pretrained.model.cls_token" in name:
        name = name.replace("core.core.pretrained.model.cls_token", "backbone.embeddings.cls_token")
    if "norm1" in name and "patch_transformer" not in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name and "patch_transformer" not in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "gamma_1" in name:
        name = name.replace("gamma_1", "lambda_1")
    if "gamma_2" in name:
        name = name.replace("gamma_2", "lambda_2")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn.relative_position_bias_table" in name:
        name = name.replace(
            "attn.relative_position_bias_table",
            "attention.attention.relative_position_bias.relative_position_bias_table",
        )
    if "attn.relative_position_index" in name:
        name = name.replace(
            "attn.relative_position_index", "attention.attention.relative_position_bias.relative_position_index"
        )

    if "core.core.pretrained.act_postprocess1.0.project" in name:
        name = name.replace(
            "core.core.pretrained.act_postprocess1.0.project", "neck.reassemble_stage.readout_projects.0"
        )
    if "core.core.pretrained.act_postprocess2.0.project" in name:
        name = name.replace(
            "core.core.pretrained.act_postprocess2.0.project", "neck.reassemble_stage.readout_projects.1"
        )
    if "core.core.pretrained.act_postprocess3.0.project" in name:
        name = name.replace(
            "core.core.pretrained.act_postprocess3.0.project", "neck.reassemble_stage.readout_projects.2"
        )
    if "core.core.pretrained.act_postprocess4.0.project" in name:
        name = name.replace(
            "core.core.pretrained.act_postprocess4.0.project", "neck.reassemble_stage.readout_projects.3"
        )

    if "core.core.pretrained.act_postprocess1.3" in name:
        name = name.replace("core.core.pretrained.act_postprocess1.3", "neck.reassemble_stage.layers.0.projection")
    if "core.core.pretrained.act_postprocess2.3" in name:
        name = name.replace("core.core.pretrained.act_postprocess2.3", "neck.reassemble_stage.layers.1.projection")
    if "core.core.pretrained.act_postprocess3.3" in name:
        name = name.replace("core.core.pretrained.act_postprocess3.3", "neck.reassemble_stage.layers.2.projection")
    if "core.core.pretrained.act_postprocess4.3" in name:
        name = name.replace("core.core.pretrained.act_postprocess4.3", "neck.reassemble_stage.layers.3.projection")

    if "core.core.pretrained.act_postprocess1.4" in name:
        name = name.replace("core.core.pretrained.act_postprocess1.4", "neck.reassemble_stage.layers.0.resize")
    if "core.core.pretrained.act_postprocess2.4" in name:
        name = name.replace("core.core.pretrained.act_postprocess2.4", "neck.reassemble_stage.layers.1.resize")
    if "core.core.pretrained.act_postprocess4.4" in name:
        name = name.replace("core.core.pretrained.act_postprocess4.4", "neck.reassemble_stage.layers.3.resize")

    if "core.core.scratch.layer1_rn.weight" in name:
        name = name.replace("core.core.scratch.layer1_rn.weight", "neck.convs.0.weight")
    if "core.core.scratch.layer2_rn.weight" in name:
        name = name.replace("core.core.scratch.layer2_rn.weight", "neck.convs.1.weight")
    if "core.core.scratch.layer3_rn.weight" in name:
        name = name.replace("core.core.scratch.layer3_rn.weight", "neck.convs.2.weight")
    if "core.core.scratch.layer4_rn.weight" in name:
        name = name.replace("core.core.scratch.layer4_rn.weight", "neck.convs.3.weight")

    if "core.core.scratch.refinenet1" in name:
        name = name.replace("core.core.scratch.refinenet1", "neck.fusion_stage.layers.3")
    if "core.core.scratch.refinenet2" in name:
        name = name.replace("core.core.scratch.refinenet2", "neck.fusion_stage.layers.2")
    if "core.core.scratch.refinenet3" in name:
        name = name.replace("core.core.scratch.refinenet3", "neck.fusion_stage.layers.1")
    if "core.core.scratch.refinenet4" in name:
        name = name.replace("core.core.scratch.refinenet4", "neck.fusion_stage.layers.0")

    if "resConfUnit1" in name:
        name = name.replace("resConfUnit1", "residual_layer1")

    if "resConfUnit2" in name:
        name = name.replace("resConfUnit2", "residual_layer2")

    if "conv1" in name:
        name = name.replace("conv1", "convolution1")

    if "conv2" in name and "residual_layer" in name:
        name = name.replace("conv2", "convolution2")

    if "out_conv" in name:
        name = name.replace("out_conv", "projection")

    if "core.core.scratch.output_conv.0" in name:
        name = name.replace("core.core.scratch.output_conv.0", "relative_head.conv1")

    if "core.core.scratch.output_conv.2" in name:
        name = name.replace("core.core.scratch.output_conv.2", "relative_head.conv2")

    if "core.core.scratch.output_conv.4" in name:
        name = name.replace("core.core.scratch.output_conv.4", "relative_head.conv3")

    if "patch_transformer" in name:
        name = name.replace("patch_transformer", "metric_head.patch_transformer")

    if "mlp_classifier.0" in name:
        name = name.replace("mlp_classifier.0", "metric_head.mlp_classifier.linear1")
    if "mlp_classifier.2" in name:
        name = name.replace("mlp_classifier.2", "metric_head.mlp_classifier.linear2")

    if "projectors" in name:
        name = name.replace("projectors", "metric_head.projectors")

    if "seed_bin_regressors" in name:
        name = name.replace("seed_bin_regressors", "metric_head.seed_bin_regressors")

    if "seed_bin_regressor" in name and "seed_bin_regressors" not in name:
        name = name.replace("seed_bin_regressor", "metric_head.seed_bin_regressor")

    if "seed_projector" in name:
        name = name.replace("seed_projector", "metric_head.seed_projector")

    if "_net.0" in name:
        name = name.replace("_net.0", "conv1")

    if "_net.2" in name:
        name = name.replace("_net.2", "conv2")

    if "attractors" in name:
        name = name.replace("attractors", "metric_head.attractors")

    if "conditional_log_binomial" in name:
        name = name.replace("conditional_log_binomial", "metric_head.conditional_log_binomial")

    if "conv2" in name and "metric_head" not in name and "attractors" not in name and "relative_head" not in name:
        name = name.replace("conv2", "metric_head.conv2")

    if "transformer_encoder.layers" in name:
        name = name.replace("transformer_encoder.layers", "transformer_encoder")

    return name


def read_in_q_k_v_metric_head(state_dict):
    pass


def convert_state_dict(orig_state_dict):
    for key in orig_state_dict.copy():
        val = orig_state_dict.pop(key)

        new_name = rename_key(key)
        orig_state_dict[new_name] = val

    return orig_state_dict


def remove_ignore_keys(state_dict):
    pass


def read_in_q_k_v(state_dict, config):
    hidden_size = config.backbone_config.hidden_size
    for i in range(config.backbone_config.num_hidden_layers):
        in_proj_weight = state_dict.pop(f"core.core.pretrained.model.blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"core.core.pretrained.model.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"core.core.pretrained.model.blocks.{i}.attn.v_bias")
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = v_bias


def prepare_img():
    filepath = hf_hub_download(repo_id="shariqfarooq/ZoeDepth", filename="examples/person_1.jpeg", repo_type="space")
    image = Image.open(filepath).convert("RGB")
    return image


@torch.no_grad()
def convert_zoedepth_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="ZoeD_N",
        choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK"],
        type=str,
        help="Name of the original ZoeDepth checkpoint you'd like to convert.",
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

    args = parser.parse_args()
    convert_zoedepth_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
