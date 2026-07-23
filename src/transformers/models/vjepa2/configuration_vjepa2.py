
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/vjepa2-vitl-fpc64-256")
@strict
class VJEPA2Config(PreTrainedConfig):

    model_type = "vjepa2"

    patch_size: int | list[int] | tuple[int, int] = 16
    crop_size: int = 256
    frames_per_clip: int = 64
    tubelet_size: int = 2
    hidden_size: int = 1024
    in_chans: int = 3
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    drop_path_rate: float | int = 0.0
    mlp_ratio: int | float = 4.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    attention_probs_dropout_prob: float | int = 0.0
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0
    num_pooler_layers: int = 3
    pred_hidden_size: int = 384
    pred_num_attention_heads: int = 12
    pred_num_hidden_layers: int = 12
    pred_num_mask_tokens: int = 10
    pred_zero_init_mask_tokens: bool = True
    pred_mlp_ratio: int | float = 4.0


__all__ = ["VJEPA2Config"]
