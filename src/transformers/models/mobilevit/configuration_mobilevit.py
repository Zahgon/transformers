
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/mobilenet_v2_1.0_224")
@strict
class MobileViTConfig(PreTrainedConfig):

    model_type = "mobilevit"

    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 256
    patch_size: int | list[int] | tuple[int, int] = 2
    hidden_sizes: list[int] | tuple[int, ...] = (144, 192, 240)
    neck_hidden_sizes: list[int] | tuple[int, ...] = (16, 32, 64, 96, 128, 160, 640)
    num_attention_heads: int = 4
    mlp_ratio: float = 2.0
    expand_ratio: float = 4.0
    hidden_act: str = "silu"
    conv_kernel_size: int = 3
    output_stride: int = 32
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.0
    classifier_dropout_prob: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    qkv_bias: bool = True
    aspp_out_channels: int = 256
    atrous_rates: list[int] | tuple[int, ...] = (6, 12, 18)
    aspp_dropout_prob: float | int = 0.1
    semantic_loss_ignore_index: int = 255


__all__ = ["MobileViTConfig"]
