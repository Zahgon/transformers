
import argparse
import json
import os
import pickle
from io import BytesIO
from pathlib import Path

import haiku as hk
import httpx
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    PerceiverConfig,
    PerceiverForImageClassificationConvProcessing,
    PerceiverForImageClassificationFourier,
    PerceiverForImageClassificationLearned,
    PerceiverForMaskedLM,
    PerceiverForMultimodalAutoencoding,
    PerceiverForOpticalFlow,
    PerceiverImageProcessor,
    PerceiverTokenizer,
)
from transformers.utils import logging

from ...utils import strtobool


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def prepare_img():
    url = "https://storage.googleapis.com/perceiver_io/dalmation.jpg"
    with httpx.stream("GET", url) as response:
        image = Image.open(BytesIO(response.read()))
    return image


def rename_keys(state_dict, architecture):
    for name in list(state_dict):
        param = state_dict.pop(name)

        name = name.replace("embed/embeddings", "input_preprocessor.embeddings.weight")
        if name.startswith("trainable_position_encoding/pos_embs"):
            name = name.replace(
                "trainable_position_encoding/pos_embs", "input_preprocessor.position_embeddings.weight"
            )

        name = name.replace("image_preprocessor/~/conv2_d/w", "input_preprocessor.convnet_1x1.weight")
        name = name.replace("image_preprocessor/~/conv2_d/b", "input_preprocessor.convnet_1x1.bias")
        name = name.replace(
            "image_preprocessor/~_build_network_inputs/trainable_position_encoding/pos_embs",
            "input_preprocessor.position_embeddings.position_embeddings",
        )
        name = name.replace(
            "image_preprocessor/~_build_network_inputs/position_encoding_projector/linear/w",
            "input_preprocessor.positions_projection.weight",
        )
        name = name.replace(
            "image_preprocessor/~_build_network_inputs/position_encoding_projector/linear/b",
            "input_preprocessor.positions_projection.bias",
        )

        if "counter" in name or "hidden" in name:
            continue
        name = name.replace(
            "image_preprocessor/~/conv2_d_downsample/~/conv/w", "input_preprocessor.convnet.conv.weight"
        )
        name = name.replace(
            "image_preprocessor/~/conv2_d_downsample/~/batchnorm/offset", "input_preprocessor.convnet.batchnorm.bias"
        )
        name = name.replace(
            "image_preprocessor/~/conv2_d_downsample/~/batchnorm/scale", "input_preprocessor.convnet.batchnorm.weight"
        )
        name = name.replace(
            "image_preprocessor/~/conv2_d_downsample/~/batchnorm/~/mean_ema/average",
            "input_preprocessor.convnet.batchnorm.running_mean",
        )
        name = name.replace(
            "image_preprocessor/~/conv2_d_downsample/~/batchnorm/~/var_ema/average",
            "input_preprocessor.convnet.batchnorm.running_var",
        )

        name = name.replace("image_preprocessor/patches_linear/b", "input_preprocessor.conv_after_patches.bias")
        name = name.replace("image_preprocessor/patches_linear/w", "input_preprocessor.conv_after_patches.weight")

        name = name.replace("multimodal_preprocessor/audio_mask_token/pos_embs", "input_preprocessor.mask.audio")
        name = name.replace("multimodal_preprocessor/audio_padding/pos_embs", "input_preprocessor.padding.audio")
        name = name.replace("multimodal_preprocessor/image_mask_token/pos_embs", "input_preprocessor.mask.image")
        name = name.replace("multimodal_preprocessor/image_padding/pos_embs", "input_preprocessor.padding.image")
        name = name.replace("multimodal_preprocessor/label_mask_token/pos_embs", "input_preprocessor.mask.label")
        name = name.replace("multimodal_preprocessor/label_padding/pos_embs", "input_preprocessor.padding.label")

        # multimodal autoencoding model
        name = name.replace(
            "multimodal_decoder/~/basic_decoder/cross_attention/", "decoder.decoder.decoding_cross_attention."
        )
        name = name.replace("multimodal_decoder/~decoder_query/audio_padding/pos_embs", "decoder.padding.audio")
        name = name.replace("multimodal_decoder/~decoder_query/image_padding/pos_embs", "decoder.padding.image")
        name = name.replace("multimodal_decoder/~decoder_query/label_padding/pos_embs", "decoder.padding.label")
        name = name.replace("multimodal_decoder/~/basic_decoder/output/b", "decoder.decoder.final_layer.bias")
        name = name.replace("multimodal_decoder/~/basic_decoder/output/w", "decoder.decoder.final_layer.weight")
        if architecture == "multimodal_autoencoding":
            name = name.replace(
                "classification_decoder/~/basic_decoder/~/trainable_position_encoding/pos_embs",
                "decoder.modalities.label.decoder.output_position_encodings.position_embeddings",
            )
        name = name.replace(
            "flow_decoder/~/basic_decoder/cross_attention/", "decoder.decoder.decoding_cross_attention."
        )
        name = name.replace("flow_decoder/~/basic_decoder/output/w", "decoder.decoder.final_layer.weight")
        name = name.replace("flow_decoder/~/basic_decoder/output/b", "decoder.decoder.final_layer.bias")
        name = name.replace(
            "classification_decoder/~/basic_decoder/~/trainable_position_encoding/pos_embs",
            "decoder.decoder.output_position_encodings.position_embeddings",
        )
        name = name.replace(
            "basic_decoder/~/trainable_position_encoding/pos_embs",
            "decoder.output_position_encodings.position_embeddings",
        )
        name = name.replace(
            "classification_decoder/~/basic_decoder/cross_attention/", "decoder.decoder.decoding_cross_attention."
        )
        name = name.replace("classification_decoder/~/basic_decoder/output/b", "decoder.decoder.final_layer.bias")
        name = name.replace("classification_decoder/~/basic_decoder/output/w", "decoder.decoder.final_layer.weight")
        name = name.replace("classification_decoder/~/basic_decoder/~/", "decoder.decoder.")
        name = name.replace("basic_decoder/cross_attention/", "decoder.decoding_cross_attention.")
        name = name.replace("basic_decoder/~/", "decoder.")

        name = name.replace(
            "projection_postprocessor/linear/b", "output_postprocessor.modalities.image.classifier.bias"
        )
        name = name.replace(
            "projection_postprocessor/linear/w", "output_postprocessor.modalities.image.classifier.weight"
        )
        name = name.replace(
            "classification_postprocessor/linear/b", "output_postprocessor.modalities.label.classifier.bias"
        )
        name = name.replace(
            "classification_postprocessor/linear/w", "output_postprocessor.modalities.label.classifier.weight"
        )
        name = name.replace("audio_postprocessor/linear/b", "output_postprocessor.modalities.audio.classifier.bias")
        name = name.replace("audio_postprocessor/linear/w", "output_postprocessor.modalities.audio.classifier.weight")


        name = name.replace("perceiver_encoder/~/trainable_position_encoding/pos_embs", "embeddings.latents")
        name = name.replace("encoder/~/trainable_position_encoding/pos_embs", "embeddings.latents")

        if name.startswith("perceiver_encoder/~/"):
            if "self_attention" in name:
                suffix = "self_attends."
            else:
                suffix = ""
            name = name.replace("perceiver_encoder/~/", "encoder." + suffix)
        if name.startswith("encoder/~/"):
            if "self_attention" in name:
                suffix = "self_attends."
            else:
                suffix = ""
            name = name.replace("encoder/~/", "encoder." + suffix)
        if "offset" in name:
            name = name.replace("offset", "bias")
        if "scale" in name:
            name = name.replace("scale", "weight")
        if "cross_attention" in name and "layer_norm_2" in name:
            name = name.replace("layer_norm_2", "layernorm")
        if "self_attention" in name and "layer_norm_1" in name:
            name = name.replace("layer_norm_1", "layernorm")

        if "cross_attention" in name and "layer_norm_1" in name:
            name = name.replace("layer_norm_1", "attention.self.layernorm2")
        if "cross_attention" in name and "layer_norm" in name:
            name = name.replace("layer_norm", "attention.self.layernorm1")
        if "self_attention" in name and "layer_norm" in name:
            name = name.replace("layer_norm", "attention.self.layernorm1")

        name = name.replace("-", ".")
        name = name.replace("/", ".")
        if ("cross_attention" in name or "self_attention" in name) and "mlp" not in name:
            if "linear.b" in name:
                name = name.replace("linear.b", "self.query.bias")
            if "linear.w" in name:
                name = name.replace("linear.w", "self.query.weight")
            if "linear_1.b" in name:
                name = name.replace("linear_1.b", "self.key.bias")
            if "linear_1.w" in name:
                name = name.replace("linear_1.w", "self.key.weight")
            if "linear_2.b" in name:
                name = name.replace("linear_2.b", "self.value.bias")
            if "linear_2.w" in name:
                name = name.replace("linear_2.w", "self.value.weight")
            if "linear_3.b" in name:
                name = name.replace("linear_3.b", "output.dense.bias")
            if "linear_3.w" in name:
                name = name.replace("linear_3.w", "output.dense.weight")
        if "self_attention_" in name:
            name = name.replace("self_attention_", "")
        if "self_attention" in name:
            name = name.replace("self_attention", "0")
        if "mlp" in name:
            if "linear.b" in name:
                name = name.replace("linear.b", "dense1.bias")
            if "linear.w" in name:
                name = name.replace("linear.w", "dense1.weight")
            if "linear_1.b" in name:
                name = name.replace("linear_1.b", "dense2.bias")
            if "linear_1.w" in name:
                name = name.replace("linear_1.w", "dense2.weight")

        if name[-6:] == "weight" and "embeddings" not in name:
            param = np.transpose(param)

        if "batchnorm" in name:
            param = np.squeeze(param)

        if "embedding_decoder" not in name:
            state_dict["perceiver." + name] = torch.from_numpy(param)
        else:
            state_dict[name] = torch.from_numpy(param)


@torch.no_grad()
def convert_perceiver_checkpoint(pickle_file, pytorch_dump_folder_path, architecture="MLM"):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickle_file",
        type=str,
        default=None,
        required=True,
        help="Path to local pickle file of a Perceiver checkpoint you'd like to convert.\n"
        "Given the files are in the pickle format, please be wary of passing it files you trust.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory, provided as a string.",
    )
    parser.add_argument(
        "--architecture",
        default="MLM",
        type=str,
        help="""
        Architecture, provided as a string. One of 'MLM', 'image_classification', image_classification_fourier',
        image_classification_fourier', 'optical_flow' or 'multimodal_autoencoding'.
        """,
    )

    args = parser.parse_args()
    convert_perceiver_checkpoint(args.pickle_file, args.pytorch_dump_folder_path, args.architecture)
