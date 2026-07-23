
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="dg845/univnet-dev")
@strict
class UnivNetConfig(PreTrainedConfig):

    model_type = "univnet"

    model_in_channels: int = 64
    model_hidden_channels: int = 32
    num_mel_bins: int = 100
    resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 3, 3)
    resblock_stride_sizes: list[int] | tuple[int, ...] = (8, 8, 4)
    resblock_dilation_sizes: list | tuple = ((1, 3, 9, 27), (1, 3, 9, 27), (1, 3, 9, 27))
    kernel_predictor_num_blocks: int = 3
    kernel_predictor_hidden_channels: int = 64
    kernel_predictor_conv_size: int = 3
    kernel_predictor_dropout: float | int = 0.0
    initializer_range: float = 0.01
    leaky_relu_slope: float = 0.2

    def validate_architecture(self):
        pass


__all__ = ["UnivNetConfig"]
