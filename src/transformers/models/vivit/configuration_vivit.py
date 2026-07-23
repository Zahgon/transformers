
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/vivit-b-16x2-kinetics400")
@strict
class VivitConfig(PreTrainedConfig):

    model_type = "vivit"

    image_size: int | list[int] | tuple[int, int] = 224
    num_frames: int = 32
    tubelet_size: list[int] | tuple[int, ...] = (2, 16, 16)
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_fast"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-06
    qkv_bias: bool = True
    pooler_output_size: int | None = None
    pooler_act: str = "tanh"

    def __post_init__(self, **kwargs):
        self.pooler_output_size = self.pooler_output_size if self.pooler_output_size else self.hidden_size
        super().__post_init__(**kwargs)


__all__ = ["VivitConfig"]
