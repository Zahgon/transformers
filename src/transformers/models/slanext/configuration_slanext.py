

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="PaddlePaddle/SLANeXt_wired_safetensors")
@strict
class SLANeXtVisionConfig(PreTrainedConfig):

    base_config_key = "vision_config"
    hidden_size: int = 768
    output_channels: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 512
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-06
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10
    qkv_bias: bool = True
    use_abs_pos: bool = True
    use_rel_pos: bool = True
    window_size: int = 14
    global_attn_indexes: list[int] | tuple[int, ...] = (2, 5, 8, 11)
    mlp_dim: int = 3072


@auto_docstring(checkpoint="PaddlePaddle/SLANeXt_wired_safetensors")
@strict
class SLANeXtConfig(PreTrainedConfig):

    model_type = "slanext"
    sub_configs = {"vision_config": SLANeXtVisionConfig}

    vision_config: dict | SLANeXtVisionConfig | None = None
    post_conv_in_channels: int = 256
    post_conv_out_channels: int = 512
    out_channels: int = 50
    hidden_size: int = 512
    max_text_length: int = 500

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = SLANeXtVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = SLANeXtVisionConfig(**self.vision_config)
        super().__post_init__(**kwargs)


__all__ = ["SLANeXtConfig"]
