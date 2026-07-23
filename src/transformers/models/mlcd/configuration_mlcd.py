from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="DeepGlint-AI/mlcd-vit-bigG-patch14-336")
@strict
class MLCDVisionConfig(PreTrainedConfig):

    model_type = "mlcd_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 1664
    intermediate_size: int = 8192
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_groups: int = 1
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 336
    patch_size: int | list[int] | tuple[int, int] = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0


__all__ = ["MLCDVisionConfig"]
