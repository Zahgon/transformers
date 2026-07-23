
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="tue-mps/videomt-dinov2-small-ytvis2019")
@strict
class VideomtConfig(PreTrainedConfig):

    model_type = "videomt"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int | list[int] | tuple[int, int] = 640
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    mlp_ratio: int = 4
    layerscale_value: float = 1.0
    drop_path_rate: float | int = 0.0
    num_upscale_blocks: int = 2
    attention_dropout: float | int = 0.0
    use_swiglu_ffn: bool = False
    num_blocks: int = 4
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    train_num_points: int = 12544
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    num_queries: int = 200
    num_register_tokens: int = 4


__all__ = ["VideomtConfig"]
