
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

from transformers import DFineConfig, DFineForObjectDetection, RTDetrImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_d_fine_config(model_name: str) -> DFineConfig:
    pass


def load_original_state_dict(repo_id, model_name):
    directory_path = hf_hub_download(repo_id=repo_id, filename=f"{model_name}.pth")

    original_state_dict = {}
    model = torch.load(directory_path, map_location="cpu")["model"]
    for key in model:
        original_state_dict[key] = model[key]

    return original_state_dict


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"decoder.valid_mask": r"model.decoder.valid_mask",
    r"decoder.anchors": r"model.decoder.anchors",
    r"decoder.up": r"model.decoder.up",
    r"decoder.reg_scale": r"model.decoder.reg_scale",
    r"backbone.stem.stem1.conv.weight": r"model.backbone.model.embedder.stem1.convolution.weight",
    r"backbone.stem.stem2a.conv.weight": r"model.backbone.model.embedder.stem2a.convolution.weight",
    r"backbone.stem.stem2b.conv.weight": r"model.backbone.model.embedder.stem2b.convolution.weight",
    r"backbone.stem.stem3.conv.weight": r"model.backbone.model.embedder.stem3.convolution.weight",
    r"backbone.stem.stem4.conv.weight": r"model.backbone.model.embedder.stem4.convolution.weight",
    r"backbone.stem.stem1.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.stem1.normalization.\1",
    r"backbone.stem.stem2a.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.stem2a.normalization.\1",
    r"backbone.stem.stem2b.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.stem2b.normalization.\1",
    r"backbone.stem.stem3.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.stem3.normalization.\1",
    r"backbone.stem.stem4.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.stem4.normalization.\1",
    r"backbone.stem.stem1.lab.(scale|bias)": r"model.backbone.model.embedder.stem1.lab.\1",
    r"backbone.stem.stem2a.lab.(scale|bias)": r"model.backbone.model.embedder.stem2a.lab.\1",
    r"backbone.stem.stem2b.lab.(scale|bias)": r"model.backbone.model.embedder.stem2b.lab.\1",
    r"backbone.stem.stem3.lab.(scale|bias)": r"model.backbone.model.embedder.stem3.lab.\1",
    r"backbone.stem.stem4.lab.(scale|bias)": r"model.backbone.model.embedder.stem4.lab.\1",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.normalization.\4",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv1.conv.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv2.conv.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv1.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.normalization.\4",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv2.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.normalization.\4",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.0.conv.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.1.conv.weight": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.convolution.weight",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.0.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.normalization.\3",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.1.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.normalization.\3",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.0.lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.0.lab.\3",
    r"backbone.stages.(\d+).blocks.(\d+).aggregation.1.lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.aggregation.1.lab.\3",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.lab.\4",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv1.lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.lab.\4",
    r"backbone.stages.(\d+).blocks.(\d+).layers.(\d+).conv2.lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.lab.\4",
    r"backbone.stages.(\d+).downsample.lab.(scale|bias)": r"model.backbone.model.encoder.stages.\1.downsample.lab.\2",
    r"backbone.stages.(\d+).downsample.conv.weight": r"model.backbone.model.encoder.stages.\1.downsample.convolution.weight",
    r"backbone.stages.(\d+).downsample.bn.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.downsample.normalization.\2",
    r"encoder.encoder.(\d+).layers.0.self_attn.out_proj.(weight|bias)": r"model.encoder.encoder.\1.layers.0.self_attn.out_proj.\2",
    r"encoder.encoder.(\d+).layers.0.linear1.(weight|bias)": r"model.encoder.encoder.\1.layers.0.fc1.\2",
    r"encoder.encoder.(\d+).layers.0.linear2.(weight|bias)": r"model.encoder.encoder.\1.layers.0.fc2.\2",
    r"encoder.encoder.(\d+).layers.0.norm1.(weight|bias)": r"model.encoder.encoder.\1.layers.0.self_attn_layer_norm.\2",
    r"encoder.encoder.(\d+).layers.0.norm2.(weight|bias)": r"model.encoder.encoder.\1.layers.0.final_layer_norm.\2",
    r"encoder.input_proj.(\d+).conv.weight": r"model.encoder_input_proj.\1.0.weight",
    r"encoder.input_proj.(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder_input_proj.\1.1.\2",
    r"encoder.lateral_convs.(\d+).conv.weight": r"model.encoder.lateral_convs.\1.conv.weight",
    r"encoder.lateral_convs.(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.lateral_convs.\1.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv1.conv.weight": r"model.encoder.fpn_blocks.\1.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv1.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv2.0.conv1.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.conv1.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv2.0.conv2.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.conv2.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv2.1.conv.weight": r"model.encoder.fpn_blocks.\1.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv2.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv3.0.conv1.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.conv1.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv3.0.conv2.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.conv2.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv3.1.conv.weight": r"model.encoder.fpn_blocks.\1.conv3.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv3.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv4.conv.weight": r"model.encoder.fpn_blocks.\1.conv4.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv4.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv4.norm.\2",
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv1.norm.\3",
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv2.norm.\3",
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv1.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv1.norm.\3",
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv2.conv.weight",
    r"encoder.fpn_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv2.norm.\3",
    r"encoder.pan_blocks.(\d+).cv1.conv.weight": r"model.encoder.pan_blocks.\1.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv1.norm.\2",
    r"encoder.pan_blocks.(\d+).cv2.0.conv1.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep1.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.conv1.norm.\2",
    r"encoder.pan_blocks.(\d+).cv2.0.conv2.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep1.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.conv2.norm.\2",
    r"encoder.pan_blocks.(\d+).cv2.1.conv.weight": r"model.encoder.pan_blocks.\1.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv2.norm.\2",
    r"encoder.pan_blocks.(\d+).cv3.0.conv1.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep2.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.conv1.norm.\2",
    r"encoder.pan_blocks.(\d+).cv3.0.conv2.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep2.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.conv2.norm.\2",
    r"encoder.pan_blocks.(\d+).cv3.1.conv.weight": r"model.encoder.pan_blocks.\1.conv3.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv3.norm.\2",
    r"encoder.pan_blocks.(\d+).cv4.conv.weight": r"model.encoder.pan_blocks.\1.conv4.conv.weight",
    r"encoder.pan_blocks.(\d+).cv4.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv4.norm.\2",
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv1.norm.\3",
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv2.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv2.norm.\3",
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv1.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv1.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv1.norm.\3",
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.conv.weight": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv2.conv.weight",
    r"encoder.pan_blocks.(\d+).cv3.0.bottlenecks.(\d+).conv2.norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv2.norm.\3",
    r"encoder.downsample_convs.(\d+).0.cv(\d+).conv.weight": r"model.encoder.downsample_convs.\1.conv\2.conv.weight",
    r"encoder.downsample_convs.(\d+).0.cv(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.downsample_convs.\1.conv\2.norm.\3",
    r"decoder.decoder.layers.(\d+).self_attn.out_proj.(weight|bias)": r"model.decoder.layers.\1.self_attn.out_proj.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.sampling_offsets.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.sampling_offsets.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.attention_weights.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.attention_weights.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.value_proj.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.value_proj.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.output_proj.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.output_proj.\2",
    r"decoder.decoder.layers.(\d+).cross_attn.num_points_scale": r"model.decoder.layers.\1.encoder_attn.num_points_scale",
    r"decoder.decoder.layers.(\d+).gateway.gate.(weight|bias)": r"model.decoder.layers.\1.gateway.gate.\2",
    r"decoder.decoder.layers.(\d+).gateway.norm.(weight|bias)": r"model.decoder.layers.\1.gateway.norm.\2",
    r"decoder.decoder.layers.(\d+).norm1.(weight|bias)": r"model.decoder.layers.\1.self_attn_layer_norm.\2",
    r"decoder.decoder.layers.(\d+).norm2.(weight|bias)": r"model.decoder.layers.\1.encoder_attn_layer_norm.\2",
    r"decoder.decoder.layers.(\d+).norm3.(weight|bias)": r"model.decoder.layers.\1.final_layer_norm.\2",
    r"decoder.decoder.layers.(\d+).linear1.(weight|bias)": r"model.decoder.layers.\1.fc1.\2",
    r"decoder.decoder.layers.(\d+).linear2.(weight|bias)": r"model.decoder.layers.\1.fc2.\2",
    r"decoder.decoder.lqe_layers.(\d+).reg_conf.layers.(\d+).(weight|bias)": r"model.decoder.lqe_layers.\1.reg_conf.layers.\2.\3",
    r"decoder.dec_score_head.(\d+).(weight|bias)": r"model.decoder.class_embed.\1.\2",
    r"decoder.dec_bbox_head.(\d+).layers.(\d+).(weight|bias)": r"model.decoder.bbox_embed.\1.layers.\2.\3",
    r"decoder.pre_bbox_head.layers.(\d+).(weight|bias)": r"model.decoder.pre_bbox_head.layers.\1.\2",
    r"decoder.input_proj.(\d+).conv.weight": r"model.decoder_input_proj.\1.0.weight",
    r"decoder.input_proj.(\d+).norm.(weight|bias|running_mean|running_var)": r"model.decoder_input_proj.\1.1.\2",
    r"decoder.denoising_class_embed.weight": r"model.denoising_class_embed.weight",
    r"decoder.query_pos_head.layers.(\d+).(weight|bias)": r"model.decoder.query_pos_head.layers.\1.\2",
    r"decoder.enc_output.proj.(weight|bias)": r"model.enc_output.0.\1",
    r"decoder.enc_output.norm.(weight|bias)": r"model.enc_output.1.\1",
    r"decoder.enc_score_head.(weight|bias)": r"model.enc_score_head.\1",
    r"decoder.enc_bbox_head.layers.(\d+).(weight|bias)": r"model.enc_bbox_head.layers.\1.\2",
}


def convert_old_keys_to_new_keys(state_dict_keys: dict | None = None):
    for original_key, converted_key in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        for key in list(state_dict_keys.keys()):
            new_key = re.sub(original_key, converted_key, key)
            if new_key != key:
                state_dict_keys[new_key] = state_dict_keys.pop(key)

    return state_dict_keys


def read_in_q_k_v(state_dict, config, model_name):
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
        in_proj_weight = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_weight", None)
        in_proj_bias = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_bias", None)
        if model_name in ["dfine_n_coco", "dfine_n_obj2coco_e25", "dfine_n_obj365"]:
            state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:128, :]
            state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:128]
            state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[128:256, :]
            state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[128:256]
            state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-128:, :]
            state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-128:]
        else:
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
def convert_d_fine_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, repo_id):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="dfine_s_coco",
        type=str,
        help="model_name of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="repo_id where the model will be pushed to.",
    )
    args = parser.parse_args()
    convert_d_fine_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
