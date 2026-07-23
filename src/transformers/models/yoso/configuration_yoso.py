
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="uw-madison/yoso-4096")
@strict
class YosoConfig(PreTrainedConfig):

    model_type = "yoso"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 4096
    type_vocab_size: int = 1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_expectation: bool = True
    hash_code_len: int = 9
    num_hash: int = 64
    conv_window: int | None = None
    use_fast_hash: bool = True
    lsh_backward: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["YosoConfig"]
