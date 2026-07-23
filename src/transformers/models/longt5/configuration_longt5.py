
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/long-t5-local-base")
@strict
class LongT5Config(PreTrainedConfig):

    model_type = "longt5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "head_dim": "d_kv",
    }

    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: int | None = None
    num_heads: int = 8
    local_radius: int = 127
    global_block_size: int = 16
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float | int = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    is_encoder_decoder: bool = True
    encoder_attention_type: str = "local"
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = None
    is_decoder: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.num_decoder_layers = self.num_decoder_layers if self.num_decoder_layers is not None else self.num_layers
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if self.feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["LongT5Config"]
