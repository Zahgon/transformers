
import argparse
import json
import re
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import RTDetrImageProcessor, RTDetrV2Config, RTDetrV2ForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_rt_detr_v2_config(model_name: str) -> RTDetrV2Config:
    pass


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"backbone.conv1.conv1_1.conv.weight": r"model.backbone.model.embedder.embedder.0.convolution.weight",
    r"backbone.conv1.conv1_1.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.embedder.0.normalization.\1",
    r"backbone.conv1.conv1_2.conv.weight": r"model.backbone.model.embedder.embedder.1.convolution.weight",
    r"backbone.conv1.conv1_2.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.embedder.1.normalization.\1",
    r"backbone.conv1.conv1_3.conv.weight": r"model.backbone.model.embedder.embedder.2.convolution.weight",
    r"backbone.conv1.conv1_3.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.embedder.2.normalization.\1",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2a.conv.weight": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.0.convolution.weight",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2a.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.0.normalization.\3",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2b.conv.weight": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.1.convolution.weight",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2b.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.1.normalization.\3",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2c.conv.weight": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.2.convolution.weight",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2c.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.2.normalization.\3",
    r"encoder.encoder.(\d+).layers.0.self_attn.out_proj.weight": r"model.encoder.encoder.\1.layers.0.self_attn.out_proj.weight",
    r"encoder.encoder.(\d+).layers.0.self_attn.out_proj.bias": r"model.encoder.encoder.\1.layers.0.self_attn.out_proj.bias",
    r"encoder.encoder.(\d+).layers.0.linear1.weight": r"model.encoder.encoder.\1.layers.0.fc1.weight",
    r"encoder.encoder.(\d+).layers.0.linear1.bias": r"model.encoder.encoder.\1.layers.0.fc1.bias",
    r"encoder.encoder.(\d+).layers.0.linear2.weight": r"model.encoder.encoder.\1.layers.0.fc2.weight",
    r"encoder.encoder.(\d+).layers.0.linear2.bias": r"model.encoder.encoder.\1.layers.0.fc2.bias",
    r"encoder.encoder.(\d+).layers.0.norm1.weight": r"model.encoder.encoder.\1.layers.0.self_attn_layer_norm.weight",
    r"encoder.encoder.(\d+).layers.0.norm1.bias": r"model.encoder.encoder.\1.layers.0.self_attn_layer_norm.bias",
    r"encoder.encoder.(\d+).layers.0.norm2.weight": r"model.encoder.encoder.\1.layers.0.final_layer_norm.weight",
    r"encoder.encoder.(\d+).layers.0.norm2.bias": r"model.encoder.encoder.\1.layers.0.final_layer_norm.bias",
    r"encoder.input_proj.(\d+).conv.weight": r"model.encoder_input_proj.\1.0.weight",
    r"encoder.input_proj.(\d+).norm.(.*)": r"model.encoder_input_proj.\1.1.\2",
    r"encoder.fpn_blocks.(\d+).conv(\d+).conv.weight": r"model.encoder.fpn_blocks.\1.conv\2.conv.weight",
    r"encoder.fpn_blocks.(\d+).conv(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv\2.norm.\3",
    r"encoder.lateral_convs.(\d+).conv.weight": r"model.encoder.lateral_convs.\1.conv.weight",
    r"encoder.lateral_convs.(\d+).norm.(.*)": r"model.encoder.lateral_convs.\1.norm.\2",
    r"encoder.fpn_blocks.(\d+).bottlenecks.(\d+).conv(\d+).conv.weight": r"model.encoder.fpn_blocks.\1.bottlenecks.\2.conv\3.conv.weight",
    r"encoder.fpn_blocks.(\d+).bottlenecks.(\d+).conv(\d+).norm.(\w+)": r"model.encoder.fpn_blocks.\1.bottlenecks.\2.conv\3.norm.\4",
    r"encoder.pan_blocks.(\d+).conv(\d+).conv.weight": r"model.encoder.pan_blocks.\1.conv\2.conv.weight",
    r"encoder.pan_blocks.(\d+).conv(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv\2.norm.\3",
    r"encoder.pan_blocks.(\d+).bottlenecks.(\d+).conv(\d+).conv.weight": r"model.encoder.pan_blocks.\1.bottlenecks.\2.conv\3.conv.weight",
    r"encoder.pan_blocks.(\d+).bottlenecks.(\d+).conv(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.bottlenecks.\2.conv\3.norm.\4",
    r"encoder.downsample_convs.(\d+).conv.weight": r"model.encoder.downsample_convs.\1.conv.weight",
    r"encoder.downsample_convs.(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.downsample_convs.\1.norm.\2",
    r"decoder.decoder.layers.(\d+).self_attn.out_proj.weight": r"model.decoder.layers.\1.self_attn.out_proj.weight",
    r"decoder.decoder.layers.(\d+).self_attn.out_proj.bias": r"model.decoder.layers.\1.self_attn.out_proj.bias",
    r"decoder.decoder.layers.(\d+).cross_attn.sampling_offsets.weight": r"model.decoder.layers.\1.encoder_attn.sampling_offsets.weight",
    r"decoder.decoder.layers.(\d+).cross_attn.sampling_offsets.bias": r"model.decoder.layers.\1.encoder_attn.sampling_offsets.bias",
    r"decoder.decoder.layers.(\d+).cross_attn.attention_weights.weight": r"model.decoder.layers.\1.encoder_attn.attention_weights.weight",
    r"decoder.decoder.layers.(\d+).cross_attn.attention_weights.bias": r"model.decoder.layers.\1.encoder_attn.attention_weights.bias",
    r"decoder.decoder.layers.(\d+).cross_attn.value_proj.weight": r"model.decoder.layers.\1.encoder_attn.value_proj.weight",
    r"decoder.decoder.layers.(\d+).cross_attn.value_proj.bias": r"model.decoder.layers.\1.encoder_attn.value_proj.bias",
    r"decoder.decoder.layers.(\d+).cross_attn.output_proj.weight": r"model.decoder.layers.\1.encoder_attn.output_proj.weight",
    r"decoder.decoder.layers.(\d+).cross_attn.output_proj.bias": r"model.decoder.layers.\1.encoder_attn.output_proj.bias",
    r"decoder.decoder.layers.(\d+).norm1.weight": r"model.decoder.layers.\1.self_attn_layer_norm.weight",
    r"decoder.decoder.layers.(\d+).norm1.bias": r"model.decoder.layers.\1.self_attn_layer_norm.bias",
    r"decoder.decoder.layers.(\d+).norm2.weight": r"model.decoder.layers.\1.encoder_attn_layer_norm.weight",
    r"decoder.decoder.layers.(\d+).norm2.bias": r"model.decoder.layers.\1.encoder_attn_layer_norm.bias",
    r"decoder.decoder.layers.(\d+).linear1.weight": r"model.decoder.layers.\1.fc1.weight",
    r"decoder.decoder.layers.(\d+).linear1.bias": r"model.decoder.layers.\1.fc1.bias",
    r"decoder.decoder.layers.(\d+).linear2.weight": r"model.decoder.layers.\1.fc2.weight",
    r"decoder.decoder.layers.(\d+).linear2.bias": r"model.decoder.layers.\1.fc2.bias",
    r"decoder.decoder.layers.(\d+).norm3.weight": r"model.decoder.layers.\1.final_layer_norm.weight",
    r"decoder.decoder.layers.(\d+).norm3.bias": r"model.decoder.layers.\1.final_layer_norm.bias",
    r"decoder.decoder.layers.(\d+).cross_attn.num_points_scale": r"model.decoder.layers.\1.encoder_attn.n_points_scale",
    r"decoder.dec_score_head.(\d+).weight": r"model.decoder.class_embed.\1.weight",
    r"decoder.dec_score_head.(\d+).bias": r"model.decoder.class_embed.\1.bias",
    r"decoder.dec_bbox_head.(\d+).layers.(\d+).(weight|bias)": r"model.decoder.bbox_embed.\1.layers.\2.\3",
    r"decoder.denoising_class_embed.weight": r"model.denoising_class_embed.weight",
    r"decoder.query_pos_head.layers.0.weight": r"model.decoder.query_pos_head.layers.0.weight",
    r"decoder.query_pos_head.layers.0.bias": r"model.decoder.query_pos_head.layers.0.bias",
    r"decoder.query_pos_head.layers.1.weight": r"model.decoder.query_pos_head.layers.1.weight",
    r"decoder.query_pos_head.layers.1.bias": r"model.decoder.query_pos_head.layers.1.bias",
    r"decoder.enc_output.proj.weight": r"model.enc_output.0.weight",
    r"decoder.enc_output.proj.bias": r"model.enc_output.0.bias",
    r"decoder.enc_output.norm.weight": r"model.enc_output.1.weight",
    r"decoder.enc_output.norm.bias": r"model.enc_output.1.bias",
    r"decoder.enc_score_head.weight": r"model.enc_score_head.weight",
    r"decoder.enc_score_head.bias": r"model.enc_score_head.bias",
    r"decoder.enc_bbox_head.layers.(\d+).(weight|bias)": r"model.enc_bbox_head.layers.\1.\2",
    r"backbone.res_layers.0.blocks.0.short.conv.weight": r"model.backbone.model.encoder.stages.0.layers.0.shortcut.convolution.weight",
    r"backbone.res_layers.0.blocks.0.short.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.0.layers.0.shortcut.normalization.\1",
    r"backbone.res_layers.(\d+).blocks.0.short.conv.conv.weight": r"model.backbone.model.encoder.stages.\1.layers.0.shortcut.1.convolution.weight",
    r"backbone.res_layers.(\d+).blocks.0.short.conv.norm.(\w+)": r"model.backbone.model.encoder.stages.\1.layers.0.shortcut.1.normalization.\2",
    r"backbone.res_layers.(\d+).blocks.0.short.conv.weight": r"model.backbone.model.encoder.stages.\1.layers.0.shortcut.1.convolution.weight",
    r"backbone.res_layers.(\d+).blocks.0.short.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.layers.0.shortcut.1.normalization.\2",
    r"decoder.input_proj.(\d+).conv.weight": r"model.decoder_input_proj.\1.0.weight",
    r"decoder.input_proj.(\d+).norm.(.*)": r"model.decoder_input_proj.\1.1.\2",
}


def convert_old_keys_to_new_keys(state_dict_keys: dict | None = None):
    for original_key, converted_key in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        for key in list(state_dict_keys.keys()):
            new_key = re.sub(original_key, converted_key, key)
            if new_key != key:
                state_dict_keys[new_key] = state_dict_keys.pop(key)

    return state_dict_keys


def read_in_q_k_v(state_dict, config):
    prefix = ""
    encoder_hidden_dim = config.encoder_hidden_dim

    for i in range(config.encoder_layers):
        in_proj_weight = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_bias")
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.weight"] = in_proj_weight[
            :encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.bias"] = in_proj_bias[:encoder_hidden_dim]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.weight"] = in_proj_weight[
            encoder_hidden_dim : 2 * encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.bias"] = in_proj_bias[
            encoder_hidden_dim : 2 * encoder_hidden_dim
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.weight"] = in_proj_weight[
            -encoder_hidden_dim:, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.bias"] = in_proj_bias[-encoder_hidden_dim:]
    for i in range(config.decoder_layers):
        in_proj_weight = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_bias")
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def write_model_and_image_processor(model_name, output_dir, push_to_hub, repo_id):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="rtdetr_v2_r18vd",
        type=str,
        help="model_name of the checkpoint you'd like to convert.",
    )
    parser.add_argument("--output_dir", default=None, type=str, help="Location to write HF model and image processor")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="repo_id where the model will be pushed to.",
    )
    args = parser.parse_args()
    write_model_and_image_processor(args.model_name, args.output_dir, args.push_to_hub, args.repo_id)
