
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/markuplm-base")
@strict
class MarkupLMConfig(PreTrainedConfig):

    model_type = "markuplm"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    max_xpath_tag_unit_embeddings: int = 256
    max_xpath_subs_unit_embeddings: int = 1024
    tag_pad_id: int = 216
    subs_pad_id: int = 1001
    xpath_unit_hidden_size: int = 32
    max_depth: int = 50
    use_cache: bool = True
    classifier_dropout: float | int | None = None


__all__ = ["MarkupLMConfig"]
