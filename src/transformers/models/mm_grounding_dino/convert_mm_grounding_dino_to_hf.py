import argparse
import re
from io import BytesIO

import httpx
import torch
from PIL import Image

from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.grounding_dino.image_processing_grounding_dino import GroundingDinoImageProcessor
from transformers.models.grounding_dino.processing_grounding_dino import GroundingDinoProcessor
from transformers.models.mm_grounding_dino.configuration_mm_grounding_dino import MMGroundingDinoConfig
from transformers.models.mm_grounding_dino.modeling_mm_grounding_dino import MMGroundingDinoForObjectDetection
from transformers.models.swin.configuration_swin import SwinConfig


MODEL_NAME_TO_CHECKPOINT_URL_MAPPING = {
    "mm_grounding_dino_tiny_o365v1_goldg": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth",
    "mm_grounding_dino_tiny_o365v1_goldg_grit": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_20231128_200818-169cc352.pth",
    "mm_grounding_dino_tiny_o365v1_goldg_v3det": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth",
    "mm_grounding_dino_tiny_o365v1_goldg_grit_v3det": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth",
    "mm_grounding_dino_base_o365v1_goldg_v3det": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth",
    "mm_grounding_dino_base_all": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth",
    "mm_grounding_dino_large_o365v2_oiv6_goldg": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth",
    "mm_grounding_dino_large_all": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth",
    "llmdet_tiny": "https://huggingface.co/fushh7/LLMDet/resolve/main/tiny.pth?download=true",
    "llmdet_base": "https://huggingface.co/fushh7/LLMDet/resolve/main/base.pth?download=true",
    "llmdet_large": "https://huggingface.co/fushh7/LLMDet/resolve/main/large.pth?download=true",
}


MODEL_NAME_TO_EXPECTED_OUTPUT_MAPPING = {
    "mm_grounding_dino_tiny_o365v1_goldg": {
        "scores": torch.tensor([0.7722, 0.7584, 0.7984, 0.7163]),
        "boxes": torch.tensor(
            [
                [0.5212, 0.1594, 0.5792, 0.3895],
                [0.5424, 0.0513, 0.9996, 0.7757],
                [0.0629, 0.1526, 0.2746, 0.2447],
                [0.0091, 0.1127, 0.4945, 0.9911],
            ]
        ),
    },
    "mm_grounding_dino_tiny_o365v1_goldg_grit": {
        "scores": torch.tensor([0.7865, 0.7180, 0.7665, 0.8177]),
        "boxes": torch.tensor(
            [
                [0.0084, 0.1129, 0.4940, 0.9895],
                [0.5214, 0.1597, 0.5786, 0.3875],
                [0.5413, 0.0507, 0.9998, 0.7768],
                [0.0631, 0.1527, 0.2740, 0.2449],
            ]
        ),
    },
    "mm_grounding_dino_tiny_o365v1_goldg_v3det": {
        "scores": torch.tensor([0.5690, 0.5553, 0.6075, 0.5775]),
        "boxes": torch.tensor(
            [
                [0.5393, 0.0502, 0.9989, 0.7763],
                [0.0090, 0.1125, 0.4950, 0.9895],
                [0.5207, 0.1589, 0.5794, 0.3889],
                [0.0625, 0.1519, 0.2750, 0.2446],
            ]
        ),
    },
    "mm_grounding_dino_tiny_o365v1_goldg_grit_v3det": {
        "scores": torch.tensor([0.8381, 0.8204, 0.7970, 0.7175]),
        "boxes": torch.tensor(
            [
                [0.0099, 0.1129, 0.4942, 0.9903],
                [0.5413, 0.0506, 0.9998, 0.7753],
                [0.0626, 0.1527, 0.2744, 0.2443],
                [0.5211, 0.1596, 0.5790, 0.3890],
            ]
        ),
    },
    "mm_grounding_dino_base_o365v1_goldg_v3det": {
        "scores": torch.tensor([0.8418, 0.8364, 0.8342, 0.7885]),
        "boxes": torch.tensor(
            [
                [0.5427, 0.0502, 0.9996, 0.7770],
                [0.0628, 0.1529, 0.2747, 0.2448],
                [0.0085, 0.1132, 0.4947, 0.9898],
                [0.5208, 0.1597, 0.5787, 0.3910],
            ]
        ),
    },
    "mm_grounding_dino_base_all": {
        "scores": torch.tensor([0.4713]),
        "boxes": torch.tensor([[0.5423, 0.0507, 0.9998, 0.7761]]),
    },
    "mm_grounding_dino_large_o365v2_oiv6_goldg": {
        "scores": torch.tensor([0.7824, 0.8275, 0.7715, 0.8211]),
        "boxes": torch.tensor(
            [
                [0.0082, 0.1133, 0.4945, 0.9889],
                [0.5410, 0.0508, 0.9998, 0.7771],
                [0.0632, 0.1526, 0.2740, 0.2439],
                [0.5205, 0.1599, 0.5787, 0.3906],
            ]
        ),
    },
    "mm_grounding_dino_large_all": {
        "scores": torch.tensor([0.7373, 0.6208, 0.6913, 0.4523]),
        "boxes": torch.tensor(
            [
                [0.5424, 0.0509, 0.9997, 0.7765],
                [0.0632, 0.1529, 0.2744, 0.2447],
                [0.0121, 0.1125, 0.4947, 0.9884],
                [0.5206, 0.1597, 0.5789, 0.3933],
            ]
        ),
    },
    "llmdet_tiny": {
        "scores": torch.tensor([0.7262, 0.7552, 0.7656, 0.8207]),
        "boxes": torch.tensor(
            [
                [0.0114, 0.1132, 0.4947, 0.9854],
                [0.5387, 0.0513, 0.9992, 0.7765],
                [0.5212, 0.1605, 0.5788, 0.3890],
                [0.0634, 0.1536, 0.2743, 0.2440],
            ]
        ),
    },
    "llmdet_base": {
        "scores": torch.tensor([0.8646, 0.7567, 0.6978, 0.8084]),
        "boxes": torch.tensor(
            [
                [0.0632, 0.1529, 0.2745, 0.2438],
                [0.5420, 0.0512, 0.9989, 0.7774],
                [0.0110, 0.1134, 0.4950, 0.9875],
                [0.5209, 0.1602, 0.5789, 0.3908],
            ]
        ),
    },
    "llmdet_large": {
        "scores": torch.tensor([0.7107, 0.8626, 0.7458, 0.8166]),
        "boxes": torch.tensor(
            [
                [0.0147, 0.1128, 0.4957, 0.9858],
                [0.0634, 0.1528, 0.2744, 0.2447],
                [0.5414, 0.0511, 0.9997, 0.7776],
                [0.5209, 0.1602, 0.5792, 0.3916],
            ]
        ),
    },
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"backbone.patch_embed.projection.(weight|bias)":                                                               r"model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection.\1",
    r"backbone.patch_embed.norm.(weight|bias)":                                                                     r"model.backbone.conv_encoder.model.embeddings.norm.\1",
    r"backbone.stages.(\d+).blocks.(\d+).attn.w_msa.(relative_position_bias_table|relative_position_index)":        r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.attention.self.\3",
    r"backbone.stages.(\d+).blocks.(\d+).norm1.(weight|bias)":                                                      r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.layernorm_before.\3",
    r"backbone.stages.(\d+).blocks.(\d+).attn.w_msa.(query|key|value).(weight|bias)":                               r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.attention.self.\3.\4",
    r"backbone.stages.(\d+).blocks.(\d+).attn.w_msa.proj.(weight|bias)":                                            r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.attention.output.dense.\3",
    r"backbone.stages.(\d+).blocks.(\d+).norm2.(weight|bias)":                                                      r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.layernorm_after.\3",
    r"backbone.stages.(\d+).blocks.(\d+).ffn.layers.0.0.(weight|bias)":                                             r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.intermediate.dense.\3",
    r"backbone.stages.(\d+).blocks.(\d+).ffn.layers.1.(weight|bias)":                                               r"model.backbone.conv_encoder.model.encoder.layers.\1.blocks.\2.output.dense.\3",
    r"backbone.stages.(\d+).downsample.reduction.weight":                                                           r"model.backbone.conv_encoder.model.encoder.layers.\1.downsample.reduction.weight",
    r"backbone.stages.(\d+).downsample.norm.(weight|bias)":                                                         r"model.backbone.conv_encoder.model.encoder.layers.\1.downsample.norm.\2",
    r"backbone.norms.(\d+).(weight|bias)":                                                                            r"model.backbone.conv_encoder.model.hidden_states_norms.stage\1.\2",
    r"neck.convs.(\d+).conv.(weight|bias)":                                                                         r"model.input_proj_vision.\1.0.\2",
    r"neck.convs.(\d+).gn.(weight|bias)":                                                                           r"model.input_proj_vision.\1.1.\2",
    r"neck.extra_convs.(\d+).conv.(weight|bias)":                                                                   r"model.input_proj_vision.\1.0.\2",
    r"neck.extra_convs.(\d+).gn.(weight|bias)":                                                                     r"model.input_proj_vision.\1.1.\2",
    r"language_model.language_backbone.body.model.(.*)":                                                            r"model.text_backbone.\1",
    r"text_feat_map.(weight|bias)":                                                                                 r"model.text_projection.\1",
    r"encoder.fusion_layers.(\d+).gamma_v":                                                                         r"model.encoder.layers.\1.fusion_layer.vision_param",
    r"encoder.fusion_layers.(\d+).gamma_l":                                                                         r"model.encoder.layers.\1.fusion_layer.text_param",
    r"encoder.fusion_layers.(\d+).layer_norm_v.(weight|bias)":                                                      r"model.encoder.layers.\1.fusion_layer.layer_norm_vision.\2",
    r"encoder.fusion_layers.(\d+).attn.v_proj.(weight|bias)":                                                       r"model.encoder.layers.\1.fusion_layer.attn.vision_proj.\2",
    r"encoder.fusion_layers.(\d+).attn.values_v_proj.(weight|bias)":                                                r"model.encoder.layers.\1.fusion_layer.attn.values_vision_proj.\2",
    r"encoder.fusion_layers.(\d+).attn.out_v_proj.(weight|bias)":                                                   r"model.encoder.layers.\1.fusion_layer.attn.out_vision_proj.\2",
    r"encoder.fusion_layers.(\d+).layer_norm_l.(weight|bias)":                                                      r"model.encoder.layers.\1.fusion_layer.layer_norm_text.\2",
    r"encoder.fusion_layers.(\d+).attn.l_proj.(weight|bias)":                                                       r"model.encoder.layers.\1.fusion_layer.attn.text_proj.\2",
    r"encoder.fusion_layers.(\d+).attn.values_l_proj.(weight|bias)":                                                r"model.encoder.layers.\1.fusion_layer.attn.values_text_proj.\2",
    r"encoder.fusion_layers.(\d+).attn.out_l_proj.(weight|bias)":                                                   r"model.encoder.layers.\1.fusion_layer.attn.out_text_proj.\2",
    r"encoder.layers.(\d+).self_attn.(sampling_offsets|attention_weights|value_proj|output_proj).(weight|bias)":    r"model.encoder.layers.\1.deformable_layer.self_attn.\2.\3",
    r"encoder.layers.(\d+).norms.0.(weight|bias)":                                                                  r"model.encoder.layers.\1.deformable_layer.self_attn_layer_norm.\2",
    r"encoder.layers.(\d+).ffn.layers.0.0.(weight|bias)":                                                           r"model.encoder.layers.\1.deformable_layer.fc1.\2",
    r"encoder.layers.(\d+).ffn.layers.1.(weight|bias)":                                                             r"model.encoder.layers.\1.deformable_layer.fc2.\2",
    r"encoder.layers.(\d+).norms.1.(weight|bias)":                                                                  r"model.encoder.layers.\1.deformable_layer.final_layer_norm.\2",
    r"encoder.text_layers.(\d+).self_attn.attn.(query|key|value)_proj_(weight|bias)":                               r"model.encoder.layers.\1.text_enhancer_layer.self_attn.\2.\3",
    r"encoder.text_layers.(\d+).self_attn.attn.out_proj.(weight|bias)":                                             r"model.encoder.layers.\1.text_enhancer_layer.self_attn.out_proj.\2",
    r"encoder.text_layers.(\d+).norms.0.(weight|bias)":                                                             r"model.encoder.layers.\1.text_enhancer_layer.layer_norm_before.\2",
    r"encoder.text_layers.(\d+).ffn.layers.0.0.(weight|bias)":                                                      r"model.encoder.layers.\1.text_enhancer_layer.fc1.\2",
    r"encoder.text_layers.(\d+).ffn.layers.1.(weight|bias)":                                                        r"model.encoder.layers.\1.text_enhancer_layer.fc2.\2",
    r"encoder.text_layers.(\d+).norms.1.(weight|bias)":                                                             r"model.encoder.layers.\1.text_enhancer_layer.layer_norm_after.\2",
    r"encoder.bbox_head.cls_branch.bias":                                                                           r"model.encoder_output_class_embed.bias",
    r"encoder.bbox_head.reg_branch.0.(weight|bias)":                                                                r"model.encoder_output_bbox_embed.layers.0.\1",
    r"encoder.bbox_head.reg_branch.2.(weight|bias)":                                                                r"model.encoder_output_bbox_embed.layers.1.\1",
    r"encoder.bbox_head.reg_branch.4.(weight|bias)":                                                                r"model.encoder_output_bbox_embed.layers.2.\1",
    r"decoder.norm.(weight|bias)":                                                                                  r"model.decoder.layer_norm.\1",
    r"decoder.ref_point_head.layers.(\d+).(weight|bias)":                                                           r"model.decoder.reference_points_head.layers.\1.\2",
    r"decoder.layers.(\d+).self_attn.attn.(query|key|value)_proj_(weight|bias)":                                    r"model.decoder.layers.\1.self_attn.\2.\3",
    r"decoder.layers.(\d+).self_attn.attn.out_proj.(weight|bias)":                                                  r"model.decoder.layers.\1.self_attn.out_proj.\2",
    r"decoder.layers.(\d+).norms.0.(weight|bias)":                                                                  r"model.decoder.layers.\1.self_attn_layer_norm.\2",
    r"decoder.layers.(\d+).cross_attn_text.attn.(query|key|value)_proj_(weight|bias)":                              r"model.decoder.layers.\1.encoder_attn_text.\2.\3",
    r"decoder.layers.(\d+).cross_attn_text.attn.out_proj.(weight|bias)":                                            r"model.decoder.layers.\1.encoder_attn_text.out_proj.\2",
    r"decoder.layers.(\d+).norms.1.(weight|bias)":                                                                  r"model.decoder.layers.\1.encoder_attn_text_layer_norm.\2",
    r"decoder.layers.(\d+).cross_attn.(sampling_offsets|attention_weights|value_proj|output_proj).(weight|bias)":   r"model.decoder.layers.\1.encoder_attn.\2.\3",
    r"decoder.layers.(\d+).norms.2.(weight|bias)":                                                                  r"model.decoder.layers.\1.encoder_attn_layer_norm.\2",
    r"decoder.layers.(\d+).ffn.layers.0.0.(weight|bias)":                                                           r"model.decoder.layers.\1.fc1.\2",
    r"decoder.layers.(\d+).ffn.layers.1.(weight|bias)":                                                             r"model.decoder.layers.\1.fc2.\2",
    r"decoder.layers.(\d+).norms.3.(weight|bias)":                                                                  r"model.decoder.layers.\1.final_layer_norm.\2",
    r"decoder.bbox_head.cls_branches.(\d+).bias":                                                                   r"model.decoder.class_embed.\1.bias",
    r"decoder.bbox_head.reg_branches.(\d+).0.(weight|bias)":                                                        r"model.decoder.bbox_embed.\1.layers.0.\2",
    r"decoder.bbox_head.reg_branches.(\d+).2.(weight|bias)":                                                        r"model.decoder.bbox_embed.\1.layers.1.\2",
    r"decoder.bbox_head.reg_branches.(\d+).4.(weight|bias)":                                                        r"model.decoder.bbox_embed.\1.layers.2.\2",
    r"level_embed":                                                                                                 r"model.level_embed",
    r"query_embedding.weight":                                                                                      r"model.query_position_embeddings.weight",
    r"memory_trans_fc.(weight|bias)":                                                                               r"model.enc_output.\1",
    r"memory_trans_norm.(weight|bias)":                                                                             r"model.enc_output_norm.\1",
    r"bbox_head.cls_branches.(\d+).bias":                                                                           r"class_embed.\1.bias",
    r"bbox_head.reg_branches.(\d+).0.(weight|bias)":                                                                r"bbox_embed.\1.layers.0.\2",
    r"bbox_head.reg_branches.(\d+).2.(weight|bias)":                                                                r"bbox_embed.\1.layers.1.\2",
    r"bbox_head.reg_branches.(\d+).4.(weight|bias)":                                                                r"bbox_embed.\1.layers.2.\2",
}


def get_mm_grounding_dino_config(model_name: str) -> MMGroundingDinoConfig:
    pass


def get_mm_grounding_dino_processor() -> GroundingDinoProcessor:
    pass


def correct_unfold_reduction_order(x: torch.Tensor) -> torch.Tensor:
    pass


def correct_unfold_norm_order(x: torch.Tensor) -> torch.Tensor:
    pass


def preprocess_old_state(state_dict: dict, config: MMGroundingDinoConfig) -> dict:
    pass


def convert_old_keys_to_new_keys(state_dict_keys: list) -> dict:
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def convert_mm_to_hf_state(original_state: dict, hf_cfg: MMGroundingDinoConfig) -> dict:
    pass


def prepare_test_inputs():
    pass


@torch.no_grad()
def convert_mm_grounding_dino_checkpoint(
    model_name: str,
    verify_outputs: bool,
    push_to_hub: bool,
    hub_user_name: str,
) -> tuple[MMGroundingDinoConfig, dict]:
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        required=True,
        type=str,
        choices=list(MODEL_NAME_TO_CHECKPOINT_URL_MAPPING.keys()),
        help="URL to the original mm grounding dino checkpoint.",
    )
    parser.add_argument("--hub-user-name", type=str, help="User name on the huggingface hub.")
    parser.add_argument("--push-to-hub", action="store_true", help="Whether to push model to hub or not.")
    parser.add_argument(
        "--verify-outputs", action="store_true", help="Whether to verify that model output is correct or not."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_mm_grounding_dino_checkpoint(
        args.model_name,
        args.verify_outputs,
        args.push_to_hub,
        args.hub_user_name,
    )
