
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="MIT/ast-finetuned-audioset-10-10-0.4593")
@strict
class ASTConfig(PreTrainedConfig):

    model_type = "audio-spectrogram-transformer"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    patch_size: int | list[int] | tuple[int, int] = 16
    qkv_bias: bool = True
    frequency_stride: int = 10
    time_stride: int = 10
    max_length: int = 1024
    num_mel_bins: int = 128


__all__ = ["ASTConfig"]
