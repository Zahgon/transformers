
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/levit-128S")
@strict
class LevitConfig(PreTrainedConfig):

    model_type = "levit"

    image_size: int | list[int] | tuple[int, int] = 224
    num_channels: int = 3
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_sizes: list[int] | tuple[int, ...] = (128, 256, 384)
    num_attention_heads: list[int] | tuple[int, ...] = (4, 8, 12)
    depths: list[int] | tuple[int, ...] = (4, 4, 4)
    key_dim: list[int] | tuple[int, ...] = (16, 16, 16)
    drop_path_rate: int = 0
    mlp_ratio: list[int] | tuple[int, ...] = (2, 2, 2)
    attention_ratio: list[int] | tuple[int, ...] = (2, 2, 2)
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.down_ops = [
            ["Subsample", self.key_dim[0], self.hidden_sizes[0] // self.key_dim[0], 4, 2, 2],
            ["Subsample", self.key_dim[0], self.hidden_sizes[1] // self.key_dim[0], 4, 2, 2],
        ]
        super().__post_init__(**kwargs)


__all__ = ["LevitConfig"]
