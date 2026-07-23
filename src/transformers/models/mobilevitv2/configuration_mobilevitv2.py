
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="apple/mobilevitv2-1.0")
@strict
class MobileViTV2Config(PreTrainedConfig):

    model_type = "mobilevitv2"

    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 256
    patch_size: int | list[int] | tuple[int, int] = 2
    expand_ratio: float = 2.0
    hidden_act: str = "swish"
    conv_kernel_size: int = 3
    output_stride: int = 32
    classifier_dropout_prob: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    aspp_out_channels: int = 512
    atrous_rates: list[int] | tuple[int, ...] = (6, 12, 18)
    aspp_dropout_prob: float | int = 0.1
    semantic_loss_ignore_index: int = 255
    n_attn_blocks: list[int] | tuple[int, ...] = (2, 4, 3)
    base_attn_unit_dims: list[int] | tuple[int, ...] = (128, 192, 256)
    width_multiplier: float | int = 1.0
    ffn_multiplier: int = 2
    attn_dropout: float | int = 0.0
    ffn_dropout: float | int = 0.0


__all__ = ["MobileViTV2Config"]
