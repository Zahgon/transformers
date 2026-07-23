
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/distilbert-base-uncased")
@strict
class DistilBertConfig(PreTrainedConfig):

    model_type = "distilbert"
    attribute_map = {
        "hidden_size": "dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    vocab_size: int = 30522
    max_position_embeddings: int = 512
    sinusoidal_pos_embds: bool = False
    n_layers: int = 6
    n_heads: int = 12
    dim: int = 768
    hidden_dim: int = 4 * 768
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation: str = "gelu"
    initializer_range: float = 0.02
    qa_dropout: float | int = 0.1
    seq_classif_dropout: float | int = 0.2
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = None
    bos_token_id: int | None = None
    tie_word_embeddings: bool = True


__all__ = ["DistilBertConfig"]
