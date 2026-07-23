
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="FacebookAI/xlm-roberta-xl")
@strict
class XLMRobertaXLConfig(PreTrainedConfig):

    model_type = "xlm-roberta-xl"

    vocab_size: int = 250880
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    intermediate_size: int = 10240
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-05
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    use_cache: bool = True
    classifier_dropout: float | int | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["XLMRobertaXLConfig"]
