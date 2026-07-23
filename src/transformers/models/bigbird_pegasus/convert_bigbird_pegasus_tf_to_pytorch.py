
import argparse

import tensorflow as tf
import torch
from tqdm import tqdm

from transformers import BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration


INIT_COMMON = [
    ("/", "."),
    ("layer_", "layers."),
    ("kernel", "weight"),
    ("beta", "bias"),
    ("gamma", "weight"),
    ("pegasus", "model"),
]
END_COMMON = [
    (".output.dense", ".fc2"),
    ("intermediate.LayerNorm", "final_layer_norm"),
    ("intermediate.dense", "fc1"),
]

DECODER_PATTERNS = (
    INIT_COMMON
    + [
        ("attention.self.LayerNorm", "self_attn_layer_norm"),
        ("attention.output.dense", "self_attn.out_proj"),
        ("attention.self", "self_attn"),
        ("attention.encdec.LayerNorm", "encoder_attn_layer_norm"),
        ("attention.encdec_output.dense", "encoder_attn.out_proj"),
        ("attention.encdec", "encoder_attn"),
        ("key", "k_proj"),
        ("value", "v_proj"),
        ("query", "q_proj"),
        ("decoder.LayerNorm", "decoder.layernorm_embedding"),
    ]
    + END_COMMON
)

REMAINING_PATTERNS = (
    INIT_COMMON
    + [
        ("embeddings.word_embeddings", "shared.weight"),
        ("embeddings.position_embeddings", "embed_positions.weight"),
        ("attention.self.LayerNorm", "self_attn_layer_norm"),
        ("attention.output.dense", "self_attn.output"),
        ("attention.self", "self_attn.self"),
        ("encoder.LayerNorm", "encoder.layernorm_embedding"),
    ]
    + END_COMMON
)

KEYS_TO_IGNORE = [
    "encdec/key/bias",
    "encdec/query/bias",
    "encdec/value/bias",
    "self/key/bias",
    "self/query/bias",
    "self/value/bias",
    "encdec_output/dense/bias",
    "attention/output/dense/bias",
]


def rename_state_dict_key(k, patterns):
    pass


def convert_bigbird_pegasus(tf_weights: dict, config_update: dict) -> BigBirdPegasusForConditionalGeneration:
    pass


def get_tf_weights_as_numpy(path) -> dict:
    pass


def convert_bigbird_pegasus_ckpt_to_pytorch(ckpt_path: str, save_dir: str, config_update: dict):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_ckpt_path", type=str, help="passed to tf.train.list_variables")
    parser.add_argument("--save_dir", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    config_update = {}
    convert_bigbird_pegasus_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir, config_update=config_update)
