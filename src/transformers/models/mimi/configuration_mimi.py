
import math

import numpy as np
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="kyutai/mimi")
@strict
class MimiConfig(PreTrainedConfig):

    model_type = "mimi"

    sampling_rate: int = 24_000
    audio_channels: int = 1
    hidden_size: int = 512
    num_filters: int = 64
    num_residual_layers: int = 1
    upsampling_ratios: list[int] | None = None
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    use_causal_conv: bool = True
    pad_mode: str = "constant"
    compress: int = 2
    trim_right_ratio: float = 1.0
    codebook_size: int = 2048
    codebook_dim: int = 256
    num_quantizers: int = 32
    use_conv_shortcut: bool = False
    vector_quantization_hidden_dimension: int = 256
    num_semantic_quantizers: int = 1
    upsample_groups: int = 512
    num_hidden_layers: int = 8
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int | None = None
    hidden_act: str = "gelu"
    max_position_embeddings: int = 8000
    initializer_range: float = 0.02
    norm_eps: float = 1e-5
    use_cache: bool = False
    use_streaming: bool = False
    rope_parameters: RopeParameters | dict | None = None
    sliding_window: int = 250
    attention_dropout: float | int = 0.0
    layer_scale_initial_scale: float = 0.01
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.upsampling_ratios = self.upsampling_ratios if self.upsampling_ratios else [8, 6, 5, 4]
        self.codebook_dim = self.codebook_dim if self.codebook_dim is not None else self.hidden_size
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        self._frame_rate = kwargs.pop("frame_rate", None)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def encodec_frame_rate(self) -> int:
        pass

    @property
    def num_codebooks(self) -> int:
        pass

    @property
    def frame_size(self) -> int:
        pass

    @property
    def frame_rate(self) -> float:
        pass


__all__ = ["MimiConfig"]
