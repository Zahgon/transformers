
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/rembert")
@strict
class RemBertConfig(PreTrainedConfig):

    model_type = "rembert"

    vocab_size: int = 250300
    hidden_size: int = 1152
    num_hidden_layers: int = 32
    num_attention_heads: int = 18
    input_embedding_size: int = 256
    output_embedding_size: int = 1664
    intermediate_size: int = 4608
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    classifier_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 312
    eos_token_id: int | list[int] | None = 313
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = False


__all__ = ["RemBertConfig"]
