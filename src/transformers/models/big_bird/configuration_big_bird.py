
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/bigbird-roberta-base")
@strict
class BigBirdConfig(PreTrainedConfig):

    model_type = "big_bird"

    vocab_size: int = 50358
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 4096
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    sep_token_id: int | None = 66
    attention_type: str = "block_sparse"
    use_bias: bool = True
    rescale_embeddings: bool = False
    block_size: int = 64
    num_random_blocks: int = 3
    classifier_dropout: float | int | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["BigBirdConfig"]
