
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="bigscience/bloom")
@strict
class BloomConfig(PreTrainedConfig):

    model_type = "bloom"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    vocab_size: int = 250880
    hidden_size: int = 64
    n_layer: int = 2
    n_head: int = 8
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = None
    apply_residual_connection_post_layernorm: bool = False
    hidden_dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    pretraining_tp: int = 1  # TP rank used when training with megatro
    slow_but_exact: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = self.hidden_size if n_embed is None else n_embed
        super().__post_init__(**kwargs)


__all__ = ["BloomConfig"]
