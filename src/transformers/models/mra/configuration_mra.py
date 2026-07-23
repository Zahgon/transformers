
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="uw-madison/mra-base-512-4")
@strict
class MraConfig(PreTrainedConfig):

    model_type = "mra"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    block_per_row: int = 4
    approx_mode: str = "full"
    initial_prior_first_n_blocks: int = 0
    initial_prior_diagonal_n_blocks: int = 0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["MraConfig"]
