
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/udop-large")
@strict
class UdopConfig(PreTrainedConfig):

    model_type = "udop"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    vocab_size: int = 33201
    d_model: int = 1024
    d_kv: int = 64
    d_ff: int = 4096
    num_layers: int = 24
    num_decoder_layers: int | None = None
    num_heads: int = 16
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    relative_bias_args: list[dict] | None = None
    dropout_rate: float | int = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    max_2d_position_embeddings: int = 1024
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.relative_bias_args is None:
            self.relative_bias_args = [{"type": "1d"}, {"type": "horizontal"}, {"type": "vertical"}]

        self.num_decoder_layers = (
            self.num_decoder_layers if self.num_decoder_layers is not None else self.num_layers
        )  # default = symmetry

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        kwargs.pop("tie_word_embeddings", None)
        self.tie_word_embeddings = True
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["UdopConfig"]
