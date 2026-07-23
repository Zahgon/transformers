
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="alibaba-damo/mgp-str-base")
@strict
class MgpstrConfig(PreTrainedConfig):

    model_type = "mgp-str"

    image_size: list[int] | tuple[int, ...] = (32, 128)
    patch_size: int | list[int] | tuple[int, int] = 4
    num_channels: int = 3
    max_token_length: int = 27
    num_character_labels: int = 38
    num_bpe_labels: int = 50257
    num_wordpiece_labels: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: float | int = 4.0
    qkv_bias: bool = True
    distilled: bool = False
    layer_norm_eps: float = 1e-5
    drop_rate: float | int = 0.0
    attn_drop_rate: float | int = 0.0
    drop_path_rate: float | int = 0.0
    output_a3_attentions: bool = False
    initializer_range: float = 0.02


__all__ = ["MgpstrConfig"]
