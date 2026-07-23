
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="openbmb/cpm-ant-10b")
@strict
class CpmAntConfig(PreTrainedConfig):

    model_type = "cpmant"

    vocab_size: int = 30720
    hidden_size: int = 4096
    num_attention_heads: int = 32
    dim_head: int = 128
    dim_ff: int = 10240
    num_hidden_layers: int = 48
    dropout_p: float | int = 0.0
    position_bias_num_buckets: int = 512
    position_bias_max_distance: int = 2048
    eps: float = 1e-6
    init_std: float = 1.0
    prompt_types: int = 32
    prompt_length: int = 32
    segment_types: int = 32
    use_cache: bool = True
    tie_word_embeddings: bool = True


__all__ = ["CpmAntConfig"]
