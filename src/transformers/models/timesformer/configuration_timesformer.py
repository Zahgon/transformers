
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/timesformer-base-finetuned-k600")
@strict
class TimesformerConfig(PreTrainedConfig):

    model_type = "timesformer"

    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    num_frames: int = 8
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    attention_type: str = "divided_space_time"
    drop_path_rate: int = 0


__all__ = ["TimesformerConfig"]
