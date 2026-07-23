

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv5_mobile_det_safetensors")
@strict
class PPOCRV5MobileDetConfig(PreTrainedConfig):

    model_type = "pp_ocrv5_mobile_det"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    reduction: int = 4
    neck_out_channels: int = 96
    interpolate_mode: str = "nearest"
    kernel_list: list[int] | tuple[int, ...] = (3, 2, 2)
    layer_list_out_channels: list[int] | tuple[int, ...] = (12, 18, 42, 360)

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet_v3",
            default_config_kwargs={
                "scale": 0.75,
                "out_features": ["stage2", "stage3", "stage4", "stage5"],
                "out_indices": [2, 3, 4, 5],
                "divisor": 16,
            },
            **kwargs,
        )

        self.id2label = {0: "text"} if self.id2label is None else self.id2label
        super().__post_init__(**kwargs)


__all__ = ["PPOCRV5MobileDetConfig"]
