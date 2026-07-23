
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv6_tiny_rec_safetensors")
@strict
class PPOCRV6TinyRecConfig(PreTrainedConfig):

    model_type = "pp_ocrv6_tiny_rec"
    sub_configs = {"backbone_config": AutoConfig}
    backbone_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 120
    head_out_channels: int = 6625

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet_v4",
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["PPOCRV6TinyRecConfig"]
