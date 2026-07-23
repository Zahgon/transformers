

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="PaddlePaddle/SLANet_plus_safetensors")
@strict
class SLANetConfig(PreTrainedConfig):

    model_type = "slanet"

    sub_configs = {"backbone_config": AutoConfig}
    post_conv_out_channels: int = 96
    out_channels: int = 50
    hidden_size: int = 256
    max_text_length: int = 500
    backbone_config: dict | PreTrainedConfig | None = None

    hidden_act: str = "hardswish"
    csp_kernel_size: int = 5
    csp_num_blocks: int = 1

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet",
            default_config_kwargs={
                "scale": 1,
                "out_features": ["stage2", "stage3", "stage4", "stage5"],
                "out_indices": [2, 3, 4, 5],
                "divisor": 16,
            },
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["SLANetConfig"]
