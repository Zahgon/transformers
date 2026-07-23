
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/deberta-base")
@strict
class DebertaConfig(PreTrainedConfig):

    model_type = "deberta"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-7
    relative_attention: bool = False
    max_relative_positions: int = -1
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    position_biased_input: bool = True
    pos_att_type: str | list[str] | None = None
    pooler_dropout: float | int = 0.0
    pooler_hidden_act: str = "gelu"
    legacy: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.pos_att_type, str):
            self.pos_att_type = [x.strip() for x in self.pos_att_type.lower().split("|")]

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", self.hidden_size)
        super().__post_init__(**kwargs)


__all__ = ["DebertaConfig"]
