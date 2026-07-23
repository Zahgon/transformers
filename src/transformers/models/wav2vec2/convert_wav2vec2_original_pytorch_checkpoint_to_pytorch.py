
import argparse
import json
import os

import fairseq
import torch
from fairseq.data import Dictionary

from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Processor,
    logging,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForSequenceClassification


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "adapter_layer": "encoder.layers.*.adapter_layer",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
    "pooling_layer.linear": "projector",
    "pooling_layer.projection": "classifier",
}
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
    "projector",
    "classifier",
]


def read_txt_into_dict(filename):
    pass


def set_recursively(key, value, full_name, weight_type, hf_pointer):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    hf_param_name = None
    for param_key in PARAM_MAPPING:
        if full_name.endswith(param_key):
            hf_param_name = PARAM_MAPPING[full_name.split(".")[-1]]
            weight_type = "param"


    if weight_type is not None and weight_type != "param":
        if weight_type == "weight_g" and not hasattr(hf_pointer, "weight_g"):
            hf_shape = hf_pointer.parametrizations.weight.original0.shape
        elif weight_type == "weight_v" and not hasattr(hf_pointer, "weight_v"):
            hf_shape = hf_pointer.parametrizations.weight.original1.shape
        else:
            hf_shape = getattr(hf_pointer, weight_type).shape
    elif weight_type is not None and weight_type == "param":
        shape_pointer = hf_pointer
        for attribute in hf_param_name.split("."):
            shape_pointer = getattr(shape_pointer, attribute)
        hf_shape = shape_pointer.shape

        value = value[0]
    else:
        hf_shape = hf_pointer.shape

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        if hasattr(hf_pointer, "weight_g"):
            hf_pointer.weight_g.data = value
        else:
            hf_pointer.parametrizations.weight.original0.data = value
    elif weight_type == "weight_v":
        if hasattr(hf_pointer, "weight_v"):
            hf_pointer.weight_v.data = value
        else:
            hf_pointer.parametrizations.weight.original1.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "param":
        for attribute in hf_param_name.split("."):
            hf_pointer = getattr(hf_pointer, attribute)
        hf_pointer.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


def rename_dict(key, value, full_name, weight_type, hf_dict):
    hf_param_name = None
    for param_key in PARAM_MAPPING:
        if full_name.endswith(param_key):
            hf_param_name = PARAM_MAPPING[full_name.split(".")[-1]]
            weight_type = "param"

    if weight_type is not None and weight_type != "param":
        full_key = ".".join([key, weight_type])
    elif weight_type is not None and weight_type == "param":
        full_key = ".".join([key, hf_param_name])
    else:
        full_key = key

    hf_dict[full_key] = value if "lm_head" in full_key else value[0]


PARAM_MAPPING = {
    "W_a": "linear_1.weight",
    "W_b": "linear_2.weight",
    "b_a": "linear_1.bias",
    "b_b": "linear_2.bias",
    "ln_W": "norm.weight",
    "ln_b": "norm.bias",
}


def load_wav2vec2_layer(name, value, hf_model=None, hf_dict=None):
    is_used = False
    for key, mapped_key in MAPPING.items():
        mapped_key = "wav2vec2." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
        if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
            is_used = True
            if "*" in mapped_key:
                layer_index = name.split(key)[0].split(".")[-2]
                mapped_key = mapped_key.replace("*", layer_index)
            if "weight_g" in name:
                weight_type = "weight_g"
            elif "weight_v" in name:
                weight_type = "weight_v"
            elif "bias" in name:
                weight_type = "bias"
            elif "weight" in name:
                weight_type = "weight"
            else:
                weight_type = None
            if hf_dict is not None:
                rename_dict(mapped_key, value, name, weight_type, hf_dict)
            else:
                set_recursively(mapped_key, value, name, weight_type, hf_model)
            return is_used
    return is_used


def recursively_load_weights(fairseq_model, hf_model, is_headless):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    feature_extractor = hf_model.wav2vec2.feature_extractor

    for name, value in fairseq_dict.items():
        is_used = False
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            is_used = load_wav2vec2_layer(name, value, hf_model)
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    if type_id == 0:
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)


@torch.no_grad()
def convert_wav2vec2_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True, is_seq_class=False
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    parser.add_argument(
        "--is_seq_class",
        action="store_true",
        help="Whether the model to convert is a fine-tuned sequence classification model or not",
    )
    args = parser.parse_args()

    is_finetuned = not args.not_finetuned and not args.is_seq_class
    convert_wav2vec2_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.dict_path,
        is_finetuned,
        args.is_seq_class,
    )
