
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="sweetcocoa/pop2piano")
@strict
class Pop2PianoConfig(PreTrainedConfig):

    model_type = "pop2piano"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_hidden_layers": "num_layers", "hidden_size": "d_model", "num_attention_heads": "num_heads"}

    vocab_size: int = 2400
    composer_vocab_size: int = 21
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: int | None = None
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float | int = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "gated-gelu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    dense_act_fn: str = "relu"
    is_decoder: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.num_decoder_layers = self.num_decoder_layers if self.num_decoder_layers is not None else self.num_layers
        self.is_gated_act = self.feed_forward_proj.split("-")[0] == "gated"
        super().__post_init__(**kwargs)


__all__ = ["Pop2PianoConfig"]
