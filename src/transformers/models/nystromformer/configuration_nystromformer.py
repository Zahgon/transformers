
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="uw-madison/nystromformer-512")
@strict
class NystromformerConfig(PreTrainedConfig):

    model_type = "nystromformer"

    vocab_size: int = 30000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 510
    type_vocab_size: int = 2
    segment_means_seq_len: int = 64
    num_landmarks: int = 64
    conv_kernel_size: int = 65
    inv_coeff_init_option: bool = False
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["NystromformerConfig"]
