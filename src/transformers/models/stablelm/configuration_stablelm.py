
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="stabilityai/stablelm-3b-4e1t")
@strict
class StableLmConfig(PreTrainedConfig):

    model_type = "stablelm"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 50304
    intermediate_size: int = 6912
    hidden_size: int = 2560
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    layer_norm_eps: float = 1.0e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    use_qkv_bias: bool = False
    qk_layernorm: bool = False
    use_parallel_residual: bool = False
    hidden_dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 0
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.25)  # assign default for BC
        super().__post_init__(**kwargs)


__all__ = ["StableLmConfig"]
