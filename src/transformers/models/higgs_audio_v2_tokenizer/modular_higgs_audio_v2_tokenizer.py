

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ...utils.import_utils import requires
from ..xcodec.configuration_xcodec import XcodecConfig
from ..xcodec.modeling_xcodec import XcodecEuclideanCodebook, XcodecModel, XcodecPreTrainedModel


@auto_docstring(checkpoint="bosonai/higgs-audio-v2-tokenizer")
@strict
class HiggsAudioV2TokenizerConfig(XcodecConfig):

    _default_semantic_model_config_kwargs = {
        "mask_time_prob": 0.0,
    }

    target_bandwidths: list[int | float] | tuple[int | float, ...] = (0.5, 1, 1.5, 2, 4)
    sample_rate: int = 24000
    codebook_dim: int = 64
    semantic_sample_rate: int = 16000
    downsample_factor: int = 320

    @property
    def semantic_downsample_factor(self):
        pass


@requires(backends=("torchaudio",))
@auto_docstring
class HiggsAudioV2TokenizerPreTrainedModel(XcodecPreTrainedModel):
    _no_split_modules = ["HiggsAudioV2TokenizerResidualVectorQuantization", "DacResidualUnit"]
    _keys_to_ignore_on_load_unexpected = ["semantic_model.masked_spec_embed"]


class HiggsAudioV2TokenizerEuclideanCodebook(XcodecEuclideanCodebook): ...


class HiggsAudioV2TokenizerVectorQuantization(nn.Module):
    def __init__(self, config: HiggsAudioV2TokenizerConfig):
        super().__init__()
        self.codebook = HiggsAudioV2TokenizerEuclideanCodebook(config)
        self.project_in = nn.Linear(config.hidden_size, config.codebook_dim)
        self.project_out = nn.Linear(config.codebook_dim, config.hidden_size)

    def encode(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.project_in(hidden_states)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        quantize = self.codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = quantize.permute(0, 2, 1)
        return quantize


@requires(backends=("torchaudio",))
@auto_docstring(custom_intro="""The HiggsAudioV2Tokenizer neural audio codec model.""")
class HiggsAudioV2TokenizerModel(XcodecModel):
    def _extract_semantic_features(self, input_values: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.sample_rate != self.config.semantic_sample_rate:
            input_values = torchaudio.functional.resample(
                input_values, self.config.sample_rate, self.config.semantic_sample_rate
            )

        input_values = input_values[:, 0, :]
        input_values = F.pad(input_values, (160, 160))
        with torch.no_grad():
            outputs = self.semantic_model(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        stacked = torch.stack([h.to(input_values.device) for h in hidden_states], dim=1)
        semantic_features = stacked.mean(dim=1)

        if self.config.semantic_downsample_factor > 1:
            semantic_features = semantic_features[:, :: self.config.semantic_downsample_factor, :]

        return semantic_features


__all__ = ["HiggsAudioV2TokenizerConfig", "HiggsAudioV2TokenizerPreTrainedModel", "HiggsAudioV2TokenizerModel"]
