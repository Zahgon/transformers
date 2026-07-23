
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/vit-mae-base")
@strict
class ViTMAEConfig(PreTrainedConfig):

    model_type = "vit_mae"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    decoder_num_attention_heads: int = 16
    decoder_hidden_size: int = 512
    decoder_num_hidden_layers: int = 8
    decoder_intermediate_size: int = 2048
    mask_ratio: float = 0.75
    norm_pix_loss: bool = False


__all__ = ["ViTMAEConfig"]
