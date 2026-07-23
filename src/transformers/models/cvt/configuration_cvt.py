
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/cvt-13")
@strict
class CvtConfig(PreTrainedConfig):

    model_type = "cvt"

    num_channels: int = 3
    patch_sizes: list[int] | tuple[int, ...] = (7, 3, 3)
    patch_stride: list[int] | tuple[int, ...] = (4, 2, 2)
    patch_padding: list[int] | tuple[int, ...] = (2, 1, 1)
    embed_dim: list[int] | tuple[int, ...] = (64, 192, 384)
    num_heads: list[int] | tuple[int, ...] = (1, 3, 6)
    depth: list[int] | tuple[int, ...] = (1, 2, 10)
    mlp_ratio: list[float] | tuple[float, ...] = (4.0, 4.0, 4.0)
    attention_drop_rate: list[float] | tuple[float, ...] = (0.0, 0.0, 0.0)
    drop_rate: list[float] | tuple[float, ...] = (0.0, 0.0, 0.0)
    drop_path_rate: list[float] | tuple[float, ...] = (0.0, 0.0, 0.1)
    qkv_bias: list[bool] | tuple[bool, ...] = (True, True, True)
    cls_token: list[bool] | tuple[bool, ...] = (False, False, True)
    qkv_projection_method: list[str] | tuple[str, ...] = ("dw_bn", "dw_bn", "dw_bn")
    kernel_qkv: list[int] | tuple[int, ...] = (3, 3, 3)
    padding_kv: list[int] | tuple[int, ...] = (1, 1, 1)
    stride_kv: list[int] | tuple[int, ...] = (2, 2, 2)
    padding_q: list[int] | tuple[int, ...] = (1, 1, 1)
    stride_q: list[int] | tuple[int, ...] = (1, 1, 1)
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12


__all__ = ["CvtConfig"]
