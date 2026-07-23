
import math

import numpy as np
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="descript/dac_16khz")
@strict
class DacConfig(PreTrainedConfig):

    model_type = "dac"

    encoder_hidden_size: int = 64
    downsampling_ratios: list[int] | tuple[int, ...] = (2, 4, 8, 8)
    decoder_hidden_size: int = 1536
    n_codebooks: int = 9
    codebook_size: int = 1024
    codebook_dim: int = 8
    quantizer_dropout: float | int = 0.0
    commitment_loss_weight: float = 0.25
    codebook_loss_weight: float = 1.0
    sampling_rate: int = 16000

    def __post_init__(self, **kwargs):
        self.upsampling_ratios = self.downsampling_ratios[::-1]
        self.hidden_size = self.encoder_hidden_size * (2 ** len(self.downsampling_ratios))
        self.hop_length = int(np.prod(self.downsampling_ratios))
        super().__post_init__(**kwargs)

    @property
    def frame_rate(self) -> int:
        pass


__all__ = ["DacConfig"]
