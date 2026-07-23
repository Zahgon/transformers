#!/usr/bin/env python

import argparse
import io
import json
import os
import tempfile
import urllib
import warnings
from typing import Any

import torch
from huggingface_hub.utils import insecure_hashlib
from torch import nn
from tqdm import tqdm

from transformers import (
    GenerationConfig,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperTokenizerFast,
)
from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode
from transformers.utils.import_utils import is_tiktoken_available


_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}


_TOKENIZERS = {
    "multilingual": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
    "english": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken",
}


def _get_generation_config(
    is_multilingual: bool,
    num_languages: int = 100,
    openai_version: str | None = None,
) -> GenerationConfig:
    pass


def remove_ignore_keys_(state_dict):
    ignore_keys = ["layers", "blocks"]
    for k in ignore_keys:
        state_dict.pop(k, None)


WHISPER_MAPPING = {
    "blocks": "layers",
    "mlp.0": "fc1",
    "mlp.2": "fc2",
    "mlp_ln": "final_layer_norm",
    ".attn.query": ".self_attn.q_proj",
    ".attn.key": ".self_attn.k_proj",
    ".attn.value": ".self_attn.v_proj",
    ".attn_ln": ".self_attn_layer_norm",
    ".attn.out": ".self_attn.out_proj",
    ".cross_attn.query": ".encoder_attn.q_proj",
    ".cross_attn.key": ".encoder_attn.k_proj",
    ".cross_attn.value": ".encoder_attn.v_proj",
    ".cross_attn_ln": ".encoder_attn_layer_norm",
    ".cross_attn.out": ".encoder_attn.out_proj",
    "decoder.ln.": "decoder.layer_norm.",
    "encoder.ln.": "encoder.layer_norm.",
    "token_embedding": "embed_tokens",
    "encoder.positional_embedding": "encoder.embed_positions.weight",
    "decoder.positional_embedding": "decoder.embed_positions.weight",
    "ln_post": "layer_norm",
}


def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)

        print(f"{key} -> {new_key}")

        s_dict[new_key] = s_dict.pop(key)
    return s_dict


def make_linear_from_emb(emb):
    pass


def _download(url: str, root: str) -> Any:
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        model_bytes = open(download_target, "rb").read()
        if insecure_hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return torch.load(io.BytesIO(model_bytes), weights_only=True)
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True, unit_divisor=1024
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if insecure_hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not match. Please retry loading the model."
        )

    return torch.load(io.BytesIO(model_bytes), weights_only=True)


def convert_openai_whisper_to_tfms(
    checkpoint_path, pytorch_dump_folder_path
) -> tuple[WhisperForConditionalGeneration, bool, int]:
    pass


def _bpe(mergeable_ranks, token: bytes, max_rank=None) -> list[bytes]:
    pass


def convert_tiktoken_bpe_to_hf(tiktoken_url: str):
    pass


def convert_tiktoken_to_hf(
    multilingual: bool = True, num_languages: int = 100, time_precision=0.02
) -> WhisperTokenizer:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, help="Path to the downloaded checkpoints")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--convert_preprocessor",
        type=bool,
        default=False,
        help="Whether or not the preprocessor (tokenizer + feature extractor) should be converted along with the model.",
    )
    args = parser.parse_args()

    model, is_multilingual, num_languages = convert_openai_whisper_to_tfms(
        args.checkpoint_path, args.pytorch_dump_folder_path
    )

    if args.convert_preprocessor:
        try:
            if not is_tiktoken_available(with_blobfile=False):
                raise ModuleNotFoundError(
                    """`tiktoken` is not installed, use `pip install tiktoken` to convert the tokenizer"""
                )
        except Exception as e:
            print(e)
        else:
            from tiktoken.load import load_tiktoken_bpe

            tokenizer = convert_tiktoken_to_hf(is_multilingual, num_languages)
            feature_extractor = WhisperFeatureExtractor(
                feature_size=model.config.num_mel_bins,
            )
            processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
            processor.save_pretrained(args.pytorch_dump_folder_path)

            fast_tokenizer = WhisperTokenizerFast.from_pretrained(args.pytorch_dump_folder_path)
            fast_tokenizer.save_pretrained(args.pytorch_dump_folder_path, legacy_format=False)

    model.save_pretrained(args.pytorch_dump_folder_path)
