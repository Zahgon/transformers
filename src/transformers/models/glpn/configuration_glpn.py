
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="vinvino02/glpn-kitti")
@strict
class GLPNConfig(PreTrainedConfig):

    model_type = "glpn"

    num_channels: int = 3
    num_encoder_blocks: int = 4
    depths: list[int] | tuple[int, ...] = (2, 2, 2, 2)
    sr_ratios: list[int] | tuple[int, ...] = (8, 4, 2, 1)
    hidden_sizes: list[int] | tuple[int, ...] = (32, 64, 160, 256)
    patch_sizes: list[int] | tuple[int, ...] = (7, 3, 3, 3)
    strides: list[int] | tuple[int, ...] = (4, 2, 2, 2)
    num_attention_heads: list[int] | tuple[int, ...] = (1, 2, 5, 8)
    mlp_ratios: list[int] | tuple[int, ...] = (4, 4, 4, 4)
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    drop_path_rate: float | int = 0.1
    layer_norm_eps: float = 1e-6
    decoder_hidden_size: int = 64
    max_depth: int = 10
    head_in_index: int = -1


__all__ = ["GLPNConfig"]
