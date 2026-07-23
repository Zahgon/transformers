
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv5_server_det_safetensors")
@strict
class PPOCRV5ServerDetConfig(PreTrainedConfig):

    sub_configs = {"backbone_config": AutoConfig}
    model_type = "pp_ocrv5_server_det"

    interpolate_mode: str = "nearest"
    backbone_config: dict | PreTrainedConfig | None = None
    neck_out_channels: int = 256
    reduce_factor: int = 2
    intraclass_block_number: int = 4
    intraclass_block_config: dict | None = None
    scale_factor: int = 2
    scale_factor_list: list | None = None
    hidden_act: str = "relu"
    kernel_list: list | None = None
    id2label: dict[int, str] | dict[str, str] | None = None

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="hgnet_v2",
            default_config_kwargs={
                "arch": "L",
                "return_idx": [0, 1, 2, 3],
                "freeze_stem_only": True,
                "freeze_at": 0,
                "freeze_norm": True,
                "lr_mult_list": [0, 0.05, 0.05, 0.05, 0.05],
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            },
            **kwargs,
        )

        self.id2label = {0: "text"} if self.id2label is None else self.id2label
        super().__post_init__(**kwargs)


__all__ = ["PPOCRV5ServerDetConfig"]
