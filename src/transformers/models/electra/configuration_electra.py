
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/electra-small-discriminator")
@strict
class ElectraConfig(PreTrainedConfig):

    model_type = "electra"

    vocab_size: int = 30522
    embedding_size: int = 128
    hidden_size: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    summary_type: str = "first"
    summary_use_proj: bool = True
    summary_activation: str = "gelu"
    summary_last_dropout: float | int = 0.1
    pad_token_id: int | None = 0
    use_cache: bool = True
    classifier_dropout: float | int | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True


__all__ = ["ElectraConfig"]
