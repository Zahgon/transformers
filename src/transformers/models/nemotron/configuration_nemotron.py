
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="thhaus/nemotron3-8b")
@strict
class NemotronConfig(PreTrainedConfig):

    model_type = "nemotron"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 256000
    hidden_size: int = 6144
    intermediate_size: int = 24576
    num_hidden_layers: int = 32
    num_attention_heads: int = 48
    head_dim: int | None = None
    num_key_value_heads: int | None = None
    hidden_act: str = "relu2"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.0134
    norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 3
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False

    def __post_init__(self, **kwargs):
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        super().__post_init__(**kwargs)


__all__ = ["NemotronConfig"]
