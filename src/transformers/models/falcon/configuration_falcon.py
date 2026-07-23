
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="tiiuae/falcon-7b")
@strict
class FalconConfig(PreTrainedConfig):

    model_type = "falcon"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 65024
    hidden_size: int = 4544
    num_hidden_layers: int = 32
    num_attention_heads: int = 71
    num_ln_in_parallel_attn: int | None = None
    layer_norm_epsilon: float | None = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    hidden_dropout: float | int | None = 0.0
    attention_dropout: float | int | None = 0.0
    num_kv_heads: int | None = None
    alibi: bool | None = False
    new_decoder_architecture: bool | None = False
    multi_query: bool | None = True
    parallel_attn: bool | None = True
    bias: bool | None = False
    max_position_embeddings: int = 2048
    rope_parameters: RopeParameters | dict | None = None
    bos_token_id: int | None = 11
    eos_token_id: int | list[int] | None = 11
    pad_token_id: int | None = None
    ffn_hidden_size: int | None = None
    activation: str | None = "gelu"
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = self.hidden_size if n_embed is None else n_embed
        self.num_kv_heads = self.num_attention_heads if self.num_kv_heads is None else self.num_kv_heads
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 4

        super().__post_init__(**kwargs)

    @property
    def head_dim(self):
        pass

    @property
    def rotary(self):
        pass


__all__ = ["FalconConfig"]
