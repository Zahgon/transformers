
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/xmod-base")
@strict
class XmodConfig(PreTrainedConfig):

    model_type = "xmod"

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
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    use_cache: bool = True
    classifier_dropout: float | int | None = None
    pre_norm: bool = False
    adapter_reduction_factor: int = 2
    adapter_layer_norm: bool = False
    adapter_reuse_layer_norm: bool = True
    ln_before_adapter: bool = True
    languages: list[str] | tuple[str, ...] = ("en_XX",)
    default_language: str | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["XmodConfig"]
