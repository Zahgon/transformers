

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="nomic-ai/nomic-embed-text-v1.5")
@strict
class NomicBertConfig(PreTrainedConfig):

    model_type = "nomic_bert"

    vocab_size: int = 30528
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "silu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    max_position_embeddings: int = 2048
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    classifier_dropout: float | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    tie_word_embeddings = True
    default_theta = 1000.0
    rope_parameters: RopeParameters | dict | None = None
    head_dim: int | None = None

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


__all__ = ["NomicBertConfig"]
