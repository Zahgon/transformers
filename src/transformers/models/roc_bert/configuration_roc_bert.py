
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="weiweishi/roc-bert-base-zh")
@strict
class RoCBertConfig(PreTrainedConfig):

    model_type = "roc_bert"

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
    use_cache: bool = True
    pad_token_id: int | None = 0
    classifier_dropout: float | int | None = None
    enable_pronunciation: bool = True
    enable_shape: bool = True
    pronunciation_embed_dim: int = 768
    pronunciation_vocab_size: int = 910
    shape_embed_dim: int = 512
    shape_vocab_size: int = 24858
    concat_input: bool = True
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["RoCBertConfig"]
