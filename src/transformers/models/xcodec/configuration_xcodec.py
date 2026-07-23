
import math

import numpy as np
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="Manel/X-Codec")
@strict
class XcodecConfig(PreTrainedConfig):

    model_type = "xcodec"

    sub_configs = {
        "acoustic_model_config": AutoConfig,
        "semantic_model_config": AutoConfig,
    }

    _default_acoustic_model_config_kwargs = {
        "encoder_hidden_size": 64,
        "downsampling_ratios": [8, 5, 4, 2],
        "decoder_hidden_size": 1024,
        "upsampling_ratios": [8, 5, 4, 2],
        "hidden_size": 256,
    }

    _default_semantic_model_config_kwargs = {}

    target_bandwidths: list[int | float] | tuple[int | float, ...] = (0.5, 1, 1.5, 2, 4)
    sample_rate: int = 16000
    kernel_size: int = 3
    channel_ratios: list[int] | tuple[int, ...] = (1, 1)
    strides: list[int] | tuple[int, ...] = (1, 1)
    block_dilations: list[int] | tuple[int, ...] = (1, 1)
    unit_kernel_size: int = 3
    codebook_size: int = 1024
    codebook_dim: int | None = None
    initializer_range: float = 0.02
    acoustic_model_config: dict | PreTrainedConfig | None = None
    semantic_model_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.acoustic_model_config is None:
            self.acoustic_model_config = CONFIG_MAPPING["dac"](
                encoder_hidden_size=64,
                downsampling_ratios=[8, 5, 4, 2],
                decoder_hidden_size=1024,
                upsampling_ratios=[8, 5, 4, 2],
                hidden_size=256,
            )
        elif isinstance(self.acoustic_model_config, dict):
            self.acoustic_model_config["model_type"] = self.acoustic_model_config.get("model_type", "dac")
            self.acoustic_model_config = CONFIG_MAPPING[self.acoustic_model_config["model_type"]](
                **{**self._default_acoustic_model_config_kwargs, **self.acoustic_model_config}
            )

        if self.semantic_model_config is None:
            self.semantic_model_config = CONFIG_MAPPING["hubert"]()
        elif isinstance(self.semantic_model_config, dict):
            self.semantic_model_config["model_type"] = self.semantic_model_config.get("model_type", "hubert")
            self.semantic_model_config = CONFIG_MAPPING[self.semantic_model_config["model_type"]](
                **{**self._default_semantic_model_config_kwargs, **self.semantic_model_config}
            )

        if self.codebook_dim is None:
            self.codebook_dim = self.acoustic_model_config.hidden_size + self.semantic_model_config.hidden_size

        super().__post_init__(**kwargs)

    @property
    def frame_rate(self) -> int:
        pass

    @property
    def semantic_hidden_size(self) -> int:
        pass

    @property
    def hop_length(self) -> int:
        pass

    @property
    def codebook_nbits(self) -> int:
        pass

    @property
    def hidden_size(self) -> int:
        pass

    @property
    def num_quantizers(self) -> int:
        pass


__all__ = ["XcodecConfig"]
