
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="studio-ousia/luke-base")
@strict
class LukeConfig(PreTrainedConfig):

    model_type = "luke"

    vocab_size: int = 50267
    entity_vocab_size: int = 500000
    hidden_size: int = 768
    entity_emb_size: int = 256
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
    use_entity_aware_attention: bool = True
    classifier_dropout: float | int | None = None
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True


__all__ = ["LukeConfig"]
