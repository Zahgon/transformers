
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="junnyu/roformer_chinese_base")
@strict
class RoFormerConfig(PreTrainedConfig):

    model_type = "roformer"

    vocab_size: int = 50000
    embedding_size: int | None = None
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 1536
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    rotary_value: bool = False
    use_cache: bool = True
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.embedding_size = self.hidden_size if self.embedding_size is None else self.embedding_size
        super().__post_init__(**kwargs)


__all__ = ["RoFormerConfig"]
