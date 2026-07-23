
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="sail/poolformer_s12")
@strict
class PoolFormerConfig(PreTrainedConfig):

    model_type = "poolformer"

    num_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    stride: int = 16
    pool_size: int = 3
    mlp_ratio: float = 4.0
    depths: list[int] | tuple[int, ...] = (2, 2, 6, 2)
    hidden_sizes: list[int] | tuple[int, ...] = (64, 128, 320, 512)
    patch_sizes: list[int] | tuple[int, ...] = (7, 3, 3, 3)
    strides: list[int] | tuple[int, ...] = (4, 2, 2, 2)
    padding: list[int] | tuple[int, ...] = (2, 1, 1, 1)
    num_encoder_blocks: int = 4
    drop_path_rate: float | int = 0.0
    hidden_act: str = "gelu"
    use_layer_scale: bool = True
    layer_scale_init_value: float = 1e-5
    initializer_range: float = 0.02


__all__ = ["PoolFormerConfig"]
