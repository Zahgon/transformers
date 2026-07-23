
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv6_small_rec_safetensors")
@strict
class PPOCRV6SmallRecConfig(PreTrainedConfig):

    model_type = "pp_ocrv6_small_rec"
    sub_configs = {"backbone_config": AutoConfig}

    hidden_act: str = "silu"
    backbone_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 120
    mlp_ratio: float = 2.0
    depth: int = 2

    head_out_channels: int = 18714
    conv_kernel_size: list | None = None
    qkv_bias: bool = True
    num_attention_heads: int = 8
    attention_dropout: float | int = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self, **kwargs):
        if self.conv_kernel_size is None:
            self.conv_kernel_size = [1, 7]
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet_v4",
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["PPOCRV6SmallRecConfig"]
