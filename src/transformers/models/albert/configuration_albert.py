
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="albert/albert-xxlarge-v2")
@strict
class AlbertConfig(PreTrainedConfig):

    model_type = "albert"

    vocab_size: int = 30000
    embedding_size: int = 128
    hidden_size: int = 4096
    num_hidden_layers: int = 12
    num_hidden_groups: int = 1
    num_attention_heads: int = 64
    intermediate_size: int = 16384
    inner_group_num: int = 1
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: int | float = 0.0
    attention_probs_dropout_prob: int | float = 0.0
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    classifier_dropout_prob: int | float = 0.1
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 3
    tie_word_embeddings: bool = True


__all__ = ["AlbertConfig"]
