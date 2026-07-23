
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/bitnet-b1.58-2B-4T")
@strict
class BitNetConfig(PreTrainedConfig):

    model_type = "bitnet"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0

    vocab_size: int = 128256
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int | None = 5
    hidden_act: str = "relu2"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = 128001
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    rope_parameters: RopeParameters | dict | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)


__all__ = ["BitNetConfig"]
