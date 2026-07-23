
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="jinaai/jina-embeddings-v3-hf")
@strict
class JinaEmbeddingsV3Config(PreTrainedConfig):

    model_type = "jina_embeddings_v3"

    vocab_size: int = 250002
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 8194
    type_vocab_size: int = 1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    use_cache: bool = True
    classifier_dropout: float | int | None = None
    tie_word_embeddings: bool = True
    default_theta = 20000.0
    rope_parameters: RopeParameters | dict | None = None


__all__ = ["JinaEmbeddingsV3Config"]
