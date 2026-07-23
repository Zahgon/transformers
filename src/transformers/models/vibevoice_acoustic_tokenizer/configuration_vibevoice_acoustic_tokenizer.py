

import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/VibeVoice-1.5B")
@strict
class VibeVoiceAcousticTokenizerConfig(PreTrainedConfig):

    model_type = "vibevoice_acoustic_tokenizer"

    channels: int = 1
    hidden_size: int = 64
    kernel_size: int = 7
    rms_norm_eps: float = 1e-5
    layer_scale_init_value: float = 1e-6
    initializer_range: float = 1e-2
    num_filters: int = 32
    downsampling_ratios: list[int] | tuple[int, ...] = (2, 2, 4, 5, 5, 8)
    depths: list[int] | tuple[int, ...] = (3, 3, 3, 3, 3, 3, 8)
    hidden_act: str = "gelu"
    ffn_expansion: int = 4
    vae_std: float = 0.625

    @property
    def hop_length(self):
        pass

    @property
    def encoder_config(self):
        pass

    @property
    def decoder_config(self):
        pass


@auto_docstring(checkpoint="microsoft/VibeVoice-1.5B")
@strict
class VibeVoiceAcousticTokenizerEncoderConfig(VibeVoiceAcousticTokenizerConfig):

    model_type = "vibevoice_acoustic_tokenizer_encoder"
    base_config_key = "encoder_config"

    @property
    def encoder_config(self):
        pass


@auto_docstring(checkpoint="microsoft/VibeVoice-1.5B")
@strict
class VibeVoiceAcousticTokenizerDecoderConfig(VibeVoiceAcousticTokenizerConfig):

    model_type = "vibevoice_acoustic_tokenizer_decoder"
    base_config_key = "decoder_config"

    @property
    def encoder_config(self):
        pass

    @property
    def decoder_config(self):
        pass

    @property
    def upsampling_ratios(self):
        pass


__all__ = [
    "VibeVoiceAcousticTokenizerConfig",
    "VibeVoiceAcousticTokenizerEncoderConfig",
    "VibeVoiceAcousticTokenizerDecoderConfig",
]
