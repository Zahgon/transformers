
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/fnet-base")
@strict
class FNetConfig(PreTrainedConfig):

    model_type = "fnet"

    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 4
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_tpu_fourier_optimizations: bool = False
    tpu_short_seq_length: int = 512
    pad_token_id: int | None = 3
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True


__all__ = ["FNetConfig"]
