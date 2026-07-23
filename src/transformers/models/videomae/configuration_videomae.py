
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="MCG-NJU/videomae-base")
@strict
class VideoMAEConfig(PreTrainedConfig):

    model_type = "videomae"

    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    num_frames: int = 16
    tubelet_size: int = 2
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    qkv_bias: bool = True
    use_mean_pooling: bool = True
    decoder_num_attention_heads: int = 6
    decoder_hidden_size: int = 384
    decoder_num_hidden_layers: int = 4
    decoder_intermediate_size: int = 1536
    norm_pix_loss: bool = True


__all__ = ["VideoMAEConfig"]
