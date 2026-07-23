
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/timesfm-2.5-200m-transformers")
@strict
class TimesFm2_5Config(PreTrainedConfig):

    model_type = "timesfm2_5"
    keys_to_ignore_at_inference = []
    is_encoder_decoder = False

    patch_length: int = 32

    context_length: int = 16384
    horizon_length: int = 128
    num_hidden_layers: int = 20
    hidden_size: int = 1280
    intermediate_size: int = 1280
    head_dim: int = 80
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    quantiles: list[float] | tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    num_key_value_heads: int = 16
    attention_bias: bool = False
    output_quantile_len: int = 1024
    decode_index: int = 5
    use_bias: bool = False
    activation: str = "swish"
    use_continuous_quantile_head: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = True
    max_position_embeddings: int = 16384
    rope_parameters: RopeParameters | dict | None = None


__all__ = ["TimesFm2_5Config"]
