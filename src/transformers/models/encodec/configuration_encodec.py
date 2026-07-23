
import math

import numpy as np
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/encodec_24khz")
@strict
class EncodecConfig(PreTrainedConfig):

    model_type = "encodec"

    target_bandwidths: list[float] | tuple[float, ...] = (1.5, 3.0, 6.0, 12.0, 24.0)
    sampling_rate: int = 24_000
    audio_channels: int = 1
    normalize: bool = False
    chunk_length_s: int | float | None = None
    overlap: float | None = None
    hidden_size: int = 128
    num_filters: int = 32
    num_residual_layers: int = 1
    upsampling_ratios: list[int] | tuple[int, ...] = (8, 5, 4, 2)
    norm_type: str = "weight_norm"
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    use_causal_conv: bool = True
    pad_mode: str = "reflect"
    compress: int = 2
    num_lstm_layers: int = 2
    trim_right_ratio: float = 1.0
    codebook_size: int = 1024
    codebook_dim: int | None = None
    use_conv_shortcut: bool = True

    def __post_init__(self, **kwargs):
        self.codebook_dim = self.codebook_dim if self.codebook_dim is not None else self.hidden_size
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def chunk_length(self) -> int | None:
        pass

    @property
    def chunk_stride(self) -> int | None:
        pass

    @property
    def hop_length(self) -> int:
        pass

    @property
    def codebook_nbits(self) -> int:
        pass

    @property
    def frame_rate(self) -> int:
        pass

    @property
    def num_quantizers(self) -> int:
        pass


__all__ = ["EncodecConfig"]
