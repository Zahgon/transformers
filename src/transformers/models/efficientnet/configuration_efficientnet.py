
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/efficientnet-b7")
@strict
class EfficientNetConfig(PreTrainedConfig):

    model_type = "efficientnet"

    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 600
    width_coefficient: float = 2.0
    depth_coefficient: float = 3.1
    depth_divisor: int = 8
    kernel_sizes: list[int] | tuple[int, ...] = (3, 3, 5, 3, 5, 5, 3)
    in_channels: list[int] | tuple[int, ...] = (32, 16, 24, 40, 80, 112, 192)
    out_channels: list[int] | tuple[int, ...] = (16, 24, 40, 80, 112, 192, 320)
    depthwise_padding: list[int] | tuple[int, ...] = ()
    strides: list[int] | tuple[int, ...] = (1, 2, 2, 2, 1, 2, 1)
    num_block_repeats: list[int] | tuple[int, ...] = (1, 2, 2, 3, 3, 4, 1)
    expand_ratios: list[int] | tuple[int, ...] = (1, 6, 6, 6, 6, 6, 6)
    squeeze_expansion_ratio: float = 0.25
    hidden_act: str = "swish"
    hidden_dim: int = 2560
    pooling_type: str = "mean"
    initializer_range: float = 0.02
    batch_norm_eps: float = 0.001
    batch_norm_momentum: float = 0.99
    dropout_rate: float | int = 0.5
    drop_connect_rate: float | int = 0.2

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.num_hidden_layers = sum(self.num_block_repeats) * 4


__all__ = ["EfficientNetConfig"]
