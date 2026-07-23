
import argparse
import json
import re
from io import BytesIO
from pathlib import Path

import httpx
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from transformers import Deimv2Config, Deimv2ForObjectDetection, RTDetrImageProcessor
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


HGNETV2_BACKBONE_CONFIGS = {
    "B0": {
        "hidden_sizes": [128, 256, 512, 1024],
        "stem_channels": [3, 16, 16],
        "stage_in_channels": [16, 64, 256, 512],
        "stage_mid_channels": [16, 32, 64, 128],
        "stage_out_channels": [64, 256, 512, 1024],
        "stage_num_blocks": [1, 1, 2, 1],
        "stage_downsample": [False, True, True, True],
        "stage_light_block": [False, False, True, True],
        "stage_kernel_size": [3, 3, 5, 5],
        "stage_numb_of_layers": [3, 3, 3, 3],
    },
    "Atto": {
        "hidden_sizes": [64, 256, 256],
        "stem_channels": [3, 16, 16],
        "stage_in_channels": [16, 64, 256],
        "stage_mid_channels": [16, 32, 64],
        "stage_out_channels": [64, 256, 256],
        "stage_num_blocks": [1, 1, 1],
        "stage_downsample": [False, True, True],
        "stage_light_block": [False, False, True],
        "stage_kernel_size": [3, 3, 3],
        "stage_numb_of_layers": [3, 3, 3],
    },
    "Femto": {
        "hidden_sizes": [64, 256, 512],
        "stem_channels": [3, 16, 16],
        "stage_in_channels": [16, 64, 256],
        "stage_mid_channels": [16, 32, 64],
        "stage_out_channels": [64, 256, 512],
        "stage_num_blocks": [1, 1, 1],
        "stage_downsample": [False, True, True],
        "stage_light_block": [False, False, True],
        "stage_kernel_size": [3, 3, 5],
        "stage_numb_of_layers": [3, 3, 3],
    },
    "Pico": {
        "hidden_sizes": [64, 256, 512],
        "stem_channels": [3, 16, 16],
        "stage_in_channels": [16, 64, 256],
        "stage_mid_channels": [16, 32, 64],
        "stage_out_channels": [64, 256, 512],
        "stage_num_blocks": [1, 1, 2],
        "stage_downsample": [False, True, True],
        "stage_light_block": [False, False, True],
        "stage_kernel_size": [3, 3, 5],
        "stage_numb_of_layers": [3, 3, 3],
    },
}
HGNETV2_BACKBONE_CONFIGS["B1"] = HGNETV2_BACKBONE_CONFIGS["B0"]
HGNETV2_BACKBONE_CONFIGS["B2"] = HGNETV2_BACKBONE_CONFIGS["B0"]


MODEL_NAME_TO_HUB_REPO = {
    "deimv2_hgnetv2_n_coco": "Intellindust/DEIMv2_HGNetv2_N_COCO",
    "deimv2_hgnetv2_pico_coco": "Intellindust/DEIMv2_HGNetv2_PICO_COCO",
    "deimv2_hgnetv2_femto_coco": "Intellindust/DEIMv2_HGNetv2_FEMTO_COCO",
    "deimv2_hgnetv2_atto_coco": "Intellindust/DEIMv2_HGNetv2_ATTO_COCO",
    "deimv2_dinov3_s_coco": "Intellindust/DEIMv2_DINOv3_S_COCO",
    "deimv2_dinov3_m_coco": "Intellindust/DEIMv2_DINOv3_M_COCO",
    "deimv2_dinov3_l_coco": "Intellindust/DEIMv2_DINOv3_L_COCO",
    "deimv2_dinov3_x_coco": "Intellindust/DEIMv2_DINOv3_X_COCO",
}


def get_deimv2_config(model_name: str) -> Deimv2Config:
    pass


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"backbone\.stem\.(stem\w+)\.conv\.weight": r"model.conv_encoder.model.embedder.\1.convolution.weight",
    r"backbone\.stem\.(stem\w+)\.bn\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.model.embedder.\1.normalization.\2",
    r"backbone\.stem\.(stem\w+)\.lab\.(scale|bias)": r"model.conv_encoder.model.embedder.\1.lab.\2",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv\.weight": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.layers.\3.convolution.weight",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.bn\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.layers.\3.normalization.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.lab\.(scale|bias)": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.layers.\3.lab.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv1\.conv\.weight": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.convolution.weight",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv1\.bn\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.normalization.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv1\.lab\.(scale|bias)": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.layers.\3.conv1.lab.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv2\.conv\.weight": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.convolution.weight",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv2\.bn\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.normalization.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)\.conv2\.lab\.(scale|bias)": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.layers.\3.conv2.lab.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation\.(\d+)\.conv\.weight": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.aggregation.\3.convolution.weight",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation\.(\d+)\.bn\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.aggregation.\3.normalization.\4",
    r"backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation\.(\d+)\.lab\.(scale|bias)": r"model.conv_encoder.model.encoder.stages.\1.blocks.\2.aggregation.\3.lab.\4",
    r"backbone\.stages\.(\d+)\.downsample\.conv\.weight": r"model.conv_encoder.model.encoder.stages.\1.downsample.convolution.weight",
    r"backbone\.stages\.(\d+)\.downsample\.bn\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.model.encoder.stages.\1.downsample.normalization.\2",
    r"backbone\.stages\.(\d+)\.downsample\.lab\.(scale|bias)": r"model.conv_encoder.model.encoder.stages.\1.downsample.lab.\2",
    r"encoder\.input_proj\.(\d+)\.conv\.weight": r"model.conv_encoder.encoder_input_proj.\1.conv.weight",
    r"encoder\.input_proj\.(\d+)\.norm\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.encoder_input_proj.\1.norm.\2",
    r"encoder\.encoder\.(\d+)\.layers\.0\.self_attn\.out_proj\.(weight|bias)": r"model.encoder.aifi.\1.layers.0.self_attn.o_proj.\2",
    r"encoder\.encoder\.(\d+)\.layers\.0\.linear1\.(weight|bias)": r"model.encoder.aifi.\1.layers.0.mlp.layers.0.\2",
    r"encoder\.encoder\.(\d+)\.layers\.0\.linear2\.(weight|bias)": r"model.encoder.aifi.\1.layers.0.mlp.layers.1.\2",
    r"encoder\.encoder\.(\d+)\.layers\.0\.norm1\.(weight|bias)": r"model.encoder.aifi.\1.layers.0.self_attn_layer_norm.\2",
    r"encoder\.encoder\.(\d+)\.layers\.0\.norm2\.(weight|bias)": r"model.encoder.aifi.\1.layers.0.final_layer_norm.\2",
    r"encoder\.lateral_convs\.(\d+)\.conv\.weight": r"model.encoder.lateral_convs.\1.conv.weight",
    r"encoder\.lateral_convs\.(\d+)\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.lateral_convs.\1.norm.\2",
    r"encoder\.fpn_blocks\.(\d+)\.cv1\.conv\.weight": r"model.encoder.fpn_blocks.\1.conv1.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv1.norm.\2",
    r"encoder\.fpn_blocks\.(\d+)\.cv4\.conv\.weight": r"model.encoder.fpn_blocks.\1.conv2.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv4\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv2.norm.\2",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.conv1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.conv1.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.conv1.norm.\2",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv1.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv1.norm.\3",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv2.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.bottlenecks.\2.conv2.norm.\3",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.conv1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.conv1.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.conv1.norm.\2",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv1.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv1.norm.\3",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv2.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.bottlenecks.\2.conv2.norm.\3",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep1.conv2.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv2\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep1.conv2.norm.\2",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.1\.conv\.weight": r"model.encoder.fpn_blocks.\1.csp_rep2.conv2.conv.weight",
    r"encoder\.fpn_blocks\.(\d+)\.cv3\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.csp_rep2.conv2.norm.\2",
    r"encoder\.pan_blocks\.(\d+)\.cv1\.conv\.weight": r"model.encoder.pan_blocks.\1.conv1.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv1.norm.\2",
    r"encoder\.pan_blocks\.(\d+)\.cv4\.conv\.weight": r"model.encoder.pan_blocks.\1.conv2.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv4\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv2.norm.\2",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.conv1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep1.conv1.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.conv1.norm.\2",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv1.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv1.norm.\3",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv2.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.bottlenecks.\2.conv2.norm.\3",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.conv1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep2.conv1.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.conv1.norm.\2",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv1.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv1.norm.\3",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv2.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.bottlenecks.\2.conv2.norm.\3",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep1.conv2.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv2\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep1.conv2.norm.\2",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.1\.conv\.weight": r"model.encoder.pan_blocks.\1.csp_rep2.conv2.conv.weight",
    r"encoder\.pan_blocks\.(\d+)\.cv3\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.csp_rep2.conv2.norm.\2",
    r"encoder\.downsample_convs\.(\d+)\.0\.cv(\d+)\.conv\.weight": r"model.encoder.downsample_convs.\1.conv\2.conv.weight",
    r"encoder\.downsample_convs\.(\d+)\.0\.cv(\d+)\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.downsample_convs.\1.conv\2.norm.\3",
    r"decoder\.input_proj\.(\d+)\.0\.weight": r"model.decoder_input_proj.\1.conv.weight",
    r"decoder\.input_proj\.(\d+)\.1\.(weight|bias|running_mean|running_var)": r"model.decoder_input_proj.\1.norm.\2",
    r"decoder\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)": r"model.decoder.layers.\1.self_attn.o_proj.\2",
    r"decoder\.decoder\.layers\.(\d+)\.cross_attn\.sampling_offsets\.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.sampling_offsets.\2",
    r"decoder\.decoder\.layers\.(\d+)\.cross_attn\.attention_weights\.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.attention_weights.\2",
    r"decoder\.decoder\.layers\.(\d+)\.cross_attn\.value_proj\.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.value_proj.\2",
    r"decoder\.decoder\.layers\.(\d+)\.cross_attn\.output_proj\.(weight|bias)": r"model.decoder.layers.\1.encoder_attn.output_proj.\2",
    r"decoder\.decoder\.layers\.(\d+)\.cross_attn\.num_points_scale": r"model.decoder.layers.\1.encoder_attn.num_points_scale",
    r"decoder\.decoder\.layers\.(\d+)\.norm1\.scale": r"model.decoder.layers.\1.self_attn_layer_norm.weight",
    r"decoder\.decoder\.layers\.(\d+)\.norm3\.scale": r"model.decoder.layers.\1.final_layer_norm.weight",
    r"decoder\.decoder\.layers\.(\d+)\.swish_ffn\.w3\.(weight|bias)": r"model.decoder.layers.\1.mlp.down_proj.\2",
    r"decoder\.decoder\.layers\.(\d+)\.gateway\.gate\.(weight|bias)": r"model.decoder.layers.\1.gateway.gate.\2",
    r"decoder\.decoder\.layers\.(\d+)\.gateway\.norm\.scale": r"model.decoder.layers.\1.gateway.norm.weight",
    r"decoder\.decoder\.lqe_layers\.(\d+)\.reg_conf\.layers\.(\d+)\.(weight|bias)": r"model.decoder.lqe_layers.\1.reg_conf.layers.\2.\3",
    r"decoder\.dec_score_head\.(\d+)\.(weight|bias)": r"model.decoder.class_embed.\1.\2",
    r"decoder\.dec_bbox_head\.(\d+)\.layers\.(\d+)\.(weight|bias)": r"model.decoder.bbox_embed.\1.layers.\2.\3",
    r"decoder\.pre_bbox_head\.layers\.(\d+)\.(weight|bias)": r"model.decoder.pre_bbox_head.layers.\1.\2",
    r"decoder\.query_pos_head\.layers\.(\d+)\.(weight|bias)": r"model.decoder.query_pos_head.layers.\1.\2",
    r"decoder\.enc_output\.proj\.(weight|bias)": r"model.enc_output.0.\1",
    r"decoder\.enc_output\.norm\.(weight|bias)": r"model.enc_output.1.\1",
    r"decoder\.enc_score_head\.(weight|bias)": r"model.enc_score_head.\1",
    r"decoder\.enc_bbox_head\.layers\.(\d+)\.(weight|bias)": r"model.enc_bbox_head.layers.\1.\2",
    r"decoder\.denoising_class_embed\.weight": r"model.denoising_class_embed.weight",
    r"decoder\.decoder\.up": r"model.decoder.up",
    r"decoder\.decoder\.reg_scale": r"model.decoder.reg_scale",
}

LITE_ENCODER_KEY_MAPPING = {
    r"encoder\.input_proj\.(\d+)\.conv\.weight": r"model.encoder.input_proj.\1.conv.weight",
    r"encoder\.input_proj\.(\d+)\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.input_proj.\1.norm.\2",
    r"encoder\.down_sample1\.1\.weight": r"model.encoder.down_conv1.conv.weight",
    r"encoder\.down_sample1\.2\.(weight|bias|running_mean|running_var)": r"model.encoder.down_conv1.norm.\1",
    r"encoder\.down_sample2\.1\.weight": r"model.encoder.down_conv2.conv.weight",
    r"encoder\.down_sample2\.2\.(weight|bias|running_mean|running_var)": r"model.encoder.down_conv2.norm.\1",
    r"encoder\.bi_fusion\.cv\.conv\.weight": r"model.encoder.bi_fusion_conv.conv.weight",
    r"encoder\.bi_fusion\.cv\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.bi_fusion_conv.norm.\1",
    r"encoder\.fpn_block\.cv1\.conv\.weight": r"model.encoder.fpn_block.conv1.conv.weight",
    r"encoder\.fpn_block\.cv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_block.conv1.norm.\1",
    r"encoder\.fpn_block\.cv4\.conv\.weight": r"model.encoder.fpn_block.conv2.conv.weight",
    r"encoder\.fpn_block\.cv4\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_block.conv2.norm.\1",
    r"encoder\.fpn_block\.cv2\.0\.conv1\.conv\.weight": r"model.encoder.fpn_block.csp_rep1.conv1.conv.weight",
    r"encoder\.fpn_block\.cv2\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_block.csp_rep1.conv1.norm.\1",
    r"encoder\.fpn_block\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.fpn_block.csp_rep1.bottlenecks.\1.conv1.conv.weight",
    r"encoder\.fpn_block\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_block.csp_rep1.bottlenecks.\1.conv1.norm.\2",
    r"encoder\.fpn_block\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.fpn_block.csp_rep1.bottlenecks.\1.conv2.conv.weight",
    r"encoder\.fpn_block\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_block.csp_rep1.bottlenecks.\1.conv2.norm.\2",
    r"encoder\.fpn_block\.cv2\.1\.conv\.weight": r"model.encoder.fpn_block.csp_rep1.conv2.conv.weight",
    r"encoder\.fpn_block\.cv2\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_block.csp_rep1.conv2.norm.\1",
    r"encoder\.fpn_block\.cv3\.0\.conv1\.conv\.weight": r"model.encoder.fpn_block.csp_rep2.conv1.conv.weight",
    r"encoder\.fpn_block\.cv3\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_block.csp_rep2.conv1.norm.\1",
    r"encoder\.fpn_block\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.fpn_block.csp_rep2.bottlenecks.\1.conv1.conv.weight",
    r"encoder\.fpn_block\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_block.csp_rep2.bottlenecks.\1.conv1.norm.\2",
    r"encoder\.fpn_block\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.fpn_block.csp_rep2.bottlenecks.\1.conv2.conv.weight",
    r"encoder\.fpn_block\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_block.csp_rep2.bottlenecks.\1.conv2.norm.\2",
    r"encoder\.fpn_block\.cv3\.1\.conv\.weight": r"model.encoder.fpn_block.csp_rep2.conv2.conv.weight",
    r"encoder\.fpn_block\.cv3\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_block.csp_rep2.conv2.norm.\1",
    r"encoder\.pan_block\.cv1\.conv\.weight": r"model.encoder.pan_block.conv1.conv.weight",
    r"encoder\.pan_block\.cv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_block.conv1.norm.\1",
    r"encoder\.pan_block\.cv4\.conv\.weight": r"model.encoder.pan_block.conv2.conv.weight",
    r"encoder\.pan_block\.cv4\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_block.conv2.norm.\1",
    r"encoder\.pan_block\.cv2\.0\.conv1\.conv\.weight": r"model.encoder.pan_block.csp_rep1.conv1.conv.weight",
    r"encoder\.pan_block\.cv2\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_block.csp_rep1.conv1.norm.\1",
    r"encoder\.pan_block\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.pan_block.csp_rep1.bottlenecks.\1.conv1.conv.weight",
    r"encoder\.pan_block\.cv2\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_block.csp_rep1.bottlenecks.\1.conv1.norm.\2",
    r"encoder\.pan_block\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.pan_block.csp_rep1.bottlenecks.\1.conv2.conv.weight",
    r"encoder\.pan_block\.cv2\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_block.csp_rep1.bottlenecks.\1.conv2.norm.\2",
    r"encoder\.pan_block\.cv2\.1\.conv\.weight": r"model.encoder.pan_block.csp_rep1.conv2.conv.weight",
    r"encoder\.pan_block\.cv2\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_block.csp_rep1.conv2.norm.\1",
    r"encoder\.pan_block\.cv3\.0\.conv1\.conv\.weight": r"model.encoder.pan_block.csp_rep2.conv1.conv.weight",
    r"encoder\.pan_block\.cv3\.0\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_block.csp_rep2.conv1.norm.\1",
    r"encoder\.pan_block\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.conv\.weight": r"model.encoder.pan_block.csp_rep2.bottlenecks.\1.conv1.conv.weight",
    r"encoder\.pan_block\.cv3\.0\.bottlenecks\.(\d+)\.conv1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_block.csp_rep2.bottlenecks.\1.conv1.norm.\2",
    r"encoder\.pan_block\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.conv\.weight": r"model.encoder.pan_block.csp_rep2.bottlenecks.\1.conv2.conv.weight",
    r"encoder\.pan_block\.cv3\.0\.bottlenecks\.(\d+)\.conv2\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_block.csp_rep2.bottlenecks.\1.conv2.norm.\2",
    r"encoder\.pan_block\.cv3\.1\.conv\.weight": r"model.encoder.pan_block.csp_rep2.conv2.conv.weight",
    r"encoder\.pan_block\.cv3\.1\.norm\.(weight|bias|running_mean|running_var)": r"model.encoder.pan_block.csp_rep2.conv2.norm.\1",
}

DECODER_NO_GATEWAY_KEY_MAPPING = {
    r"decoder\.decoder\.layers\.(\d+)\.norm2\.scale": r"model.decoder.layers.\1.encoder_attn_layer_norm.weight",
}

DINOV3_KEY_MAPPING = {
    r"backbone\.dinov3\.patch_embed\.proj\.(weight|bias)": r"model.conv_encoder.backbone.embeddings.patch_embeddings.\1",
    r"backbone\.dinov3\.cls_token": r"model.conv_encoder.backbone.embeddings.cls_token",
    r"backbone\.dinov3\.storage_tokens": r"model.conv_encoder.backbone.embeddings.register_tokens",
    r"backbone\.dinov3\.mask_token": r"model.conv_encoder.backbone.embeddings.mask_token",
    r"backbone\.dinov3\.blocks\.(\d+)\.norm1\.(weight|bias)": r"model.conv_encoder.backbone.model.layer.\1.norm1.\2",
    r"backbone\.dinov3\.blocks\.(\d+)\.norm2\.(weight|bias)": r"model.conv_encoder.backbone.model.layer.\1.norm2.\2",
    r"backbone\.dinov3\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)": r"model.conv_encoder.backbone.model.layer.\1.attention.qkv.\2",
    r"backbone\.dinov3\.blocks\.(\d+)\.attn\.proj\.(weight|bias)": r"model.conv_encoder.backbone.model.layer.\1.attention.o_proj.\2",
    r"backbone\.dinov3\.blocks\.(\d+)\.mlp\.fc1\.(weight|bias)": r"model.conv_encoder.backbone.model.layer.\1.mlp.up_proj.\2",
    r"backbone\.dinov3\.blocks\.(\d+)\.mlp\.fc2\.(weight|bias)": r"model.conv_encoder.backbone.model.layer.\1.mlp.down_proj.\2",
    r"backbone\.dinov3\.blocks\.(\d+)\.mlp\.w1\.(weight|bias)": r"model.conv_encoder.backbone.model.layer.\1.mlp.gate_proj.\2",
    r"backbone\.dinov3\.blocks\.(\d+)\.mlp\.w2\.(weight|bias)": r"model.conv_encoder.backbone.model.layer.\1.mlp.up_proj.\2",
    r"backbone\.dinov3\.blocks\.(\d+)\.mlp\.w3\.(weight|bias)": r"model.conv_encoder.backbone.model.layer.\1.mlp.down_proj.\2",
    r"backbone\.dinov3\.blocks\.(\d+)\.ls1\.gamma": r"model.conv_encoder.backbone.model.layer.\1.layer_scale1.lambda1",
    r"backbone\.dinov3\.blocks\.(\d+)\.ls2\.gamma": r"model.conv_encoder.backbone.model.layer.\1.layer_scale2.lambda1",
    r"backbone\.dinov3\.norm\.(weight|bias)": r"model.conv_encoder.backbone.norm.\1",
    r"backbone\.sta\.stem\.0\.(weight)": r"model.conv_encoder.spatial_tuning_adapter.stem_conv.conv.\1",
    r"backbone\.sta\.stem\.1\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.spatial_tuning_adapter.stem_conv.norm.\1",
    r"backbone\.sta\.conv2\.0\.(weight)": r"model.conv_encoder.spatial_tuning_adapter.conv2.conv.\1",
    r"backbone\.sta\.conv2\.1\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.spatial_tuning_adapter.conv2.norm.\1",
    r"backbone\.sta\.conv3\.1\.(weight)": r"model.conv_encoder.spatial_tuning_adapter.conv3.conv.\1",
    r"backbone\.sta\.conv3\.2\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.spatial_tuning_adapter.conv3.norm.\1",
    r"backbone\.sta\.conv4\.1\.(weight)": r"model.conv_encoder.spatial_tuning_adapter.conv4.conv.\1",
    r"backbone\.sta\.conv4\.2\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.spatial_tuning_adapter.conv4.norm.\1",
    r"backbone\.convs\.(\d+)\.weight": r"model.conv_encoder.fusion_proj.\1.conv.weight",
    r"backbone\.norms\.(\d+)\.(weight|bias|running_mean|running_var)": r"model.conv_encoder.fusion_proj.\1.norm.\2",
}


def convert_old_keys_to_new_keys(state_dict, config=None):
    mapping = dict(ORIGINAL_TO_CONVERTED_KEY_MAPPING)

    is_dinov3 = getattr(config.backbone_config, "model_type", None) == "dinov3_vit" if config else False

    if config is not None:
        if config.encoder_type == "lite":
            for k in list(mapping.keys()):
                if (
                    k.startswith(r"encoder\.input_proj")
                    or k.startswith(r"encoder\.lateral")
                    or k.startswith(r"encoder\.fpn_blocks")
                    or k.startswith(r"encoder\.pan_blocks")
                    or k.startswith(r"encoder\.downsample")
                    or k.startswith(r"encoder\.encoder")
                ):
                    del mapping[k]
            mapping.update(LITE_ENCODER_KEY_MAPPING)

        if not config.use_gateway:
            mapping.update(DECODER_NO_GATEWAY_KEY_MAPPING)
            for k in list(mapping.keys()):
                if "gateway" in k:
                    del mapping[k]

        if is_dinov3:
            for k in list(mapping.keys()):
                if k.startswith(r"backbone\.") or k.startswith(r"encoder\.input_proj"):
                    del mapping[k]
            mapping.update(DINOV3_KEY_MAPPING)

    for original_key, converted_key in mapping.items():
        for key in list(state_dict.keys()):
            new_key = re.sub(f"^{original_key}$", converted_key, key)
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)
    return state_dict


def read_in_q_k_v(state_dict, config):
    encoder_hidden_dim = config.encoder_hidden_dim
    d_model = config.d_model

    for i in range(config.encoder_layers):
        in_proj_weight = state_dict.pop(f"encoder.encoder.{i}.layers.0.self_attn.in_proj_weight", None)
        in_proj_bias = state_dict.pop(f"encoder.encoder.{i}.layers.0.self_attn.in_proj_bias", None)
        if in_proj_weight is not None:
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.q_proj.weight"] = in_proj_weight[
                :encoder_hidden_dim
            ]
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.k_proj.weight"] = in_proj_weight[
                encoder_hidden_dim : 2 * encoder_hidden_dim
            ]
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.v_proj.weight"] = in_proj_weight[
                -encoder_hidden_dim:
            ]
        if in_proj_bias is not None:
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.q_proj.bias"] = in_proj_bias[:encoder_hidden_dim]
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.k_proj.bias"] = in_proj_bias[
                encoder_hidden_dim : 2 * encoder_hidden_dim
            ]
            state_dict[f"model.encoder.aifi.{i}.layers.0.self_attn.v_proj.bias"] = in_proj_bias[-encoder_hidden_dim:]

    for i in range(config.decoder_layers):
        in_proj_weight = state_dict.pop(f"decoder.decoder.layers.{i}.self_attn.in_proj_weight", None)
        in_proj_bias = state_dict.pop(f"decoder.decoder.layers.{i}.self_attn.in_proj_bias", None)
        if in_proj_weight is not None:
            state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:d_model]
            state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:d_model]
            state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[d_model : 2 * d_model]
            state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[d_model : 2 * d_model]
            state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-d_model:]
            state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-d_model:]


def split_swiglu_fused_weights(state_dict, config):
    pass


def strip_dinov3_model_prefix(state_dict):
    pass


def read_in_q_k_v_vit(state_dict, config):
    pass


def load_original_state_dict(repo_id):
    filepath = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    return load_file(filepath)


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


@torch.no_grad()
def convert_deimv2_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, repo_id):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="deimv2_hgnetv2_n_coco",
        type=str,
        choices=list(MODEL_NAME_TO_HUB_REPO.keys()),
        help="Model name to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push to the hub.")
    parser.add_argument("--repo_id", type=str, default=None, help="Hub repo_id to push to.")
    args = parser.parse_args()
    convert_deimv2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
