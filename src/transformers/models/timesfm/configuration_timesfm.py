
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/timesfm-2.0-500m-pytorch")
@strict
class TimesFmConfig(PreTrainedConfig):

    model_type = "timesfm"
    keys_to_ignore_at_inference = []
    is_encoder_decoder = False

    patch_length: int = 32
    context_length: int = 512
    horizon_length: int = 128
    freq_size: int = 3
    num_hidden_layers: int = 50
    hidden_size: int = 1280
    intermediate_size: int = 1280
    head_dim: int = 80
    num_attention_heads: int = 16
    tolerance: float = 1e-6
    rms_norm_eps: float = 1e-6
    quantiles: list[float] | tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    pad_val: float = 1123581321.0
    attention_dropout: float | int = 0.0
    use_positional_embedding: bool = False
    initializer_range: float = 0.02
    min_timescale: int = 1
    max_timescale: int = 10_000


__all__ = ["TimesFmConfig"]
