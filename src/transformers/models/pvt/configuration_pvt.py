
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="Xrenya/pvt-tiny-224")
@strict
class PvtConfig(PreTrainedConfig):

    model_type = "pvt"

    image_size: int | list[int] | tuple[int, int] = 224
    num_channels: int = 3
    num_encoder_blocks: int = 4
    depths: list[int] | tuple[int, ...] = (2, 2, 2, 2)
    sequence_reduction_ratios: list[int] | tuple[int, ...] = (8, 4, 2, 1)
    hidden_sizes: list[int] | tuple[int, ...] = (64, 128, 320, 512)
    patch_sizes: list[int] | tuple[int, ...] = (4, 2, 2, 2)
    strides: list[int] | tuple[int, ...] = (4, 2, 2, 2)
    num_attention_heads: list[int] | tuple[int, ...] = (1, 2, 5, 8)
    mlp_ratios: list[int] | tuple[int, ...] = (8, 8, 4, 4)
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    drop_path_rate: float | int = 0.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    num_labels: int = 1000


__all__ = ["PvtConfig"]
