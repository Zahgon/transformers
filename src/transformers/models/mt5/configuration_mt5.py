
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/mt5-small")
@strict
class MT5Config(PreTrainedConfig):

    model_type = "mt5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "head_dim": "d_kv",
    }

    vocab_size: int = 250112
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 1024
    num_layers: int = 8
    num_decoder_layers: int | None = None
    num_heads: int = 6
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float | int = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "gated-gelu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    tie_word_embeddings: bool = True
    bos_token_id: int | None = None
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    decoder_start_token_id: int | None = 0
    classifier_dropout: float | int = 0.0
    is_decoder: bool = False

    def __post_init__(self, **kwargs):
        self.num_decoder_layers = (
            self.num_decoder_layers if self.num_decoder_layers is not None else self.num_layers
        )  # default = symmetry

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if self.feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        kwargs.pop("tie_word_embeddings", None)
        self.tie_word_embeddings = True
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["MT5Config"]
