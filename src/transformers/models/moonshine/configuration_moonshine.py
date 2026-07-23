
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="UsefulSensors/moonshine-tiny")
@strict
class MoonshineConfig(PreTrainedConfig):

    model_type = "moonshine"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_key_value_heads": "decoder_num_key_value_heads",
        "num_attention_heads": "decoder_num_attention_heads",
        "num_hidden_layers": "decoder_num_hidden_layers",
        "hidden_act": "decoder_hidden_act",
    }

    vocab_size: int = 32768
    hidden_size: int = 288
    intermediate_size: int = 1152
    encoder_num_hidden_layers: int = 6
    decoder_num_hidden_layers: int = 6
    encoder_num_attention_heads: int = 8
    decoder_num_attention_heads: int = 8
    encoder_num_key_value_heads: int | None = None
    decoder_num_key_value_heads: int | None = None
    pad_head_dim_to_multiple_of: int | None = None
    encoder_hidden_act: str = "gelu"
    decoder_hidden_act: str = "silu"
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    decoder_start_token_id: int = 1
    use_cache: bool = True
    rope_parameters: RopeParameters | dict | None = None
    is_encoder_decoder: bool = True
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.encoder_num_key_value_heads is None:
            self.encoder_num_key_value_heads = self.encoder_num_attention_heads

        if self.decoder_num_key_value_heads is None:
            self.decoder_num_key_value_heads = self.decoder_num_attention_heads

        kwargs.setdefault("partial_rotary_factor", 0.9)  # assign default for BC
        super().__post_init__(**kwargs)


__all__ = ["MoonshineConfig"]
