
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/opt-350m")
@strict
class OPTConfig(PreTrainedConfig):

    model_type = "opt"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 50272
    hidden_size: int = 768
    num_hidden_layers: int = 12
    ffn_dim: int = 3072
    max_position_embeddings: int = 2048
    do_layer_norm_before: bool = True
    _remove_final_layer_norm: bool = False
    word_embed_proj_dim: int | None = None
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    num_attention_heads: int = 12
    activation_function: str = "relu"
    layerdrop: float | int = 0.0
    init_std: float = 0.02
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 2
    enable_bias: bool = True
    layer_norm_elementwise_affine: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.word_embed_proj_dim = (
            self.word_embed_proj_dim if self.word_embed_proj_dim is not None else self.hidden_size
        )
        super().__post_init__(**kwargs)


__all__ = ["OPTConfig"]
