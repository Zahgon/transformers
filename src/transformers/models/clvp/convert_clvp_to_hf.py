

import argparse
import os

import torch
from huggingface_hub import hf_hub_download

from transformers import ClvpConfig, ClvpModelForConditionalGeneration


_MODELS = {
    "clvp": "https://huggingface.co/jbetker/tortoise-tts-v2/blob/main/.models/clvp2.pth",
    "decoder": "https://huggingface.co/jbetker/tortoise-tts-v2/blob/main/.models/autoregressive.pth",
}

dim = 1024
sub_dim = dim // 16

CLVP_ENCODERS_MAPPING = {
    "text_transformer.transformer.attn_layers": "text_encoder_model",
    "speech_transformer.transformer.attn_layers": "speech_encoder_model",
    "text_transformer.transformer.norm": "text_encoder_model.final_layer_norm",
    "speech_transformer.transformer.norm": "speech_encoder_model.final_layer_norm",
    "to_text_latent": "text_encoder_model.projection",
    "to_speech_latent": "speech_encoder_model.projection",
    "text_emb": "text_encoder_model.token_embedding",
    "speech_emb": "speech_encoder_model.token_embedding",
    "1.wrap.net.0": "mlp.fc1",
    "1.wrap.net.3": "mlp.fc2",
    "1.wrap": "self_attn",
    "to_out": "out_proj",
    "to_q": "q_proj",
    "to_k": "k_proj",
    "to_v": "v_proj",
    "temperature": "logit_scale",
}

CLVP_DECODER_MAPPING = {
    "conditioning_encoder.init": "conditioning_encoder.mel_conv",
    "conditioning_encoder.attn": "conditioning_encoder.mel_attn_blocks",
    "mel_attn_blocks": "group_norms",
    ".norm.weight": ".weight",
    ".norm.bias": ".bias",
    "text_embedding": "conditioning_encoder.text_token_embedding",
    "text_pos_embedding.emb": "conditioning_encoder.text_position_embedding",
    "final_norm": "speech_decoder_model.final_norm",
    "mel_head": "speech_decoder_model.lm_head",
    "gpt.ln_f": "speech_decoder_model.model.decoder.layer_norm",
    "mel_embedding": "speech_decoder_model.model.decoder.input_embeds_layer",
    "mel_pos_embedding.emb": "speech_decoder_model.model.decoder.position_embeds_layer",
    "gpt.h": "speech_decoder_model.model.decoder.layers",
    "ln_1": "input_layernorm",
    "ln_2": "post_attention_layernorm",
}


def update_index(present_index):
    if present_index % 2 == 0:
        return int(present_index / 2)
    else:
        return int((present_index - 1) / 2)


def convert_encoder_weights(original_weights):
    pass


def convert_decoder_weights(original_weights):
    pass


def _download(url: str, root: str):
    repo_id = f"{url.split('/')[3]}/{url.split('/')[4]}"
    filename = f"{url.split('/')[-2]}/{url.split('/')[-1]}"
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_filename=root,
        local_dir_use_symlinks=False,
    )


def convert_clvp_weights(checkpoint_path, pytorch_dump_folder_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to the folder of downloaded checkpoints. (Please enter full path)"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model. (Please enter full path)",
    )
    args = parser.parse_args()

    convert_clvp_weights(args.checkpoint_path, args.pytorch_dump_folder_path)
