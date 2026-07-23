
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="MBZUAI/swiftformer-xs")
@strict
class SwiftFormerConfig(PreTrainedConfig):

    model_type = "swiftformer"

    image_size: int | list[int] | tuple[int, int] = 224
    num_channels: int = 3
    depths: list[int] | tuple[int, ...] = (3, 3, 6, 4)
    embed_dims: list[int] | tuple[int, ...] = (48, 56, 112, 220)
    mlp_ratio: int = 4
    downsamples: list[bool] | tuple[bool, ...] = (True, True, True, True)
    hidden_act: str = "gelu"
    down_patch_size: int | list[int] | tuple[int, int] = 3
    down_stride: int = 2
    down_pad: int = 1
    drop_path_rate: float | int = 0.0
    drop_mlp_rate: float | int = 0.0
    drop_conv_encoder_rate: float | int = 0.0
    use_layer_scale: bool = True
    layer_scale_init_value: float = 1e-5
    batch_norm_eps: float = 1e-5


__all__ = ["SwiftFormerConfig"]
