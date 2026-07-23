
import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from audiocraft.models import MusicGen

from transformers import (
    AutoTokenizer,
    EncodecModel,
    T5EncoderModel,
)
from transformers.models.musicgen_melody.configuration_musicgen_melody import MusicgenMelodyDecoderConfig
from transformers.models.musicgen_melody.feature_extraction_musicgen_melody import MusicgenMelodyFeatureExtractor
from transformers.models.musicgen_melody.modeling_musicgen_melody import (
    MusicgenMelodyForCausalLM,
    MusicgenMelodyForConditionalGeneration,
)
from transformers.models.musicgen_melody.processing_musicgen_melody import MusicgenMelodyProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


EXPECTED_MISSING_KEYS = ["model.decoder.embed_positions.weights"]
EXPECTED_ADDITIONAL_KEYS = ["condition_provider.conditioners.self_wav.chroma.spec.window"]


def rename_keys(name):
    if "emb" in name:
        name = name.replace("emb", "model.decoder.embed_tokens")
    if "transformer" in name:
        name = name.replace("transformer", "model.decoder")
    if "cross_attention" in name:
        name = name.replace("cross_attention", "encoder_attn")
    if "linear1" in name:
        name = name.replace("linear1", "fc1")
    if "linear2" in name:
        name = name.replace("linear2", "fc2")
    if "norm1" in name:
        name = name.replace("norm1", "self_attn_layer_norm")
    if "norm_cross" in name:
        name = name.replace("norm_cross", "encoder_attn_layer_norm")
    if "norm2" in name:
        name = name.replace("norm2", "final_layer_norm")
    if "out_norm" in name:
        name = name.replace("out_norm", "model.decoder.layer_norm")
    if "linears" in name:
        name = name.replace("linears", "lm_heads")
    if "condition_provider.conditioners.description.output_proj" in name:
        name = name.replace("condition_provider.conditioners.description.output_proj", "enc_to_dec_proj")
    if "condition_provider.conditioners.self_wav.output_proj" in name:
        name = name.replace("condition_provider.conditioners.self_wav.output_proj", "audio_enc_to_dec_proj")
    return name


def rename_state_dict(state_dict: OrderedDict, hidden_size: int) -> tuple[dict, dict]:
    """Function that takes the fairseq MusicgenMelody state dict and renames it according to the HF
    module names. It further partitions the state dict into the decoder (LM) state dict, and that for the
    text encoder projection and for the audio encoder projection."""
    keys = list(state_dict.keys())
    enc_dec_proj_state_dict = {}
    audio_enc_to_dec_proj_state_dict = {}
    for key in keys:
        val = state_dict.pop(key)
        key = rename_keys(key)
        if "in_proj_weight" in key:
            state_dict[key.replace("in_proj_weight", "q_proj.weight")] = val[:hidden_size, :]
            state_dict[key.replace("in_proj_weight", "k_proj.weight")] = val[hidden_size : 2 * hidden_size, :]
            state_dict[key.replace("in_proj_weight", "v_proj.weight")] = val[-hidden_size:, :]
        elif "audio_enc_to_dec_proj" in key:
            audio_enc_to_dec_proj_state_dict[key[len("audio_enc_to_dec_proj.") :]] = val
        elif "enc_to_dec_proj" in key:
            enc_dec_proj_state_dict[key[len("enc_to_dec_proj.") :]] = val
        else:
            state_dict[key] = val
    return state_dict, enc_dec_proj_state_dict, audio_enc_to_dec_proj_state_dict


def decoder_config_from_checkpoint(checkpoint: str) -> MusicgenMelodyDecoderConfig:
    pass


@torch.no_grad()
def convert_musicgen_melody_checkpoint(
    checkpoint, pytorch_dump_folder=None, repo_id=None, device="cpu", test_same_output=False
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="facebook/musicgen-melody",
        type=str,
        help="Checkpoint size of the Musicgen Melody model you'd like to convert. Can be one of: "
        "`['facebook/musicgen-melody', 'facebook/musicgen-melody-large']` for the mono checkpoints, or "
        "`['facebook/musicgen-stereo-melody', 'facebook/musicgen-stereo-melody-large']` "
        "for the stereo checkpoints.",
    )
    parser.add_argument(
        "--pytorch_dump_folder",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        default="musicgen-melody",
        type=str,
        help="Where to upload the converted model on the Hugging Face hub.",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Torch device to run the conversion, either cpu or cuda."
    )
    parser.add_argument("--test_same_output", default=False, type=bool, help="If `True`, test if same output logits.")

    args = parser.parse_args()
    convert_musicgen_melody_checkpoint(
        args.checkpoint, args.pytorch_dump_folder, args.push_to_hub, args.device, args.test_same_output
    )
