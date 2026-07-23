
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="openmmlab/upernet-convnext-tiny")
@strict
class UperNetConfig(PreTrainedConfig):

    model_type = "upernet"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 512
    initializer_range: float = 0.02
    pool_scales: list[int] | tuple[int, ...] = (1, 2, 3, 6)
    use_auxiliary_head: bool = True
    auxiliary_loss_weight: float = 0.4
    auxiliary_in_channels: int | None = None
    auxiliary_channels: int = 256
    auxiliary_num_convs: int = 1
    auxiliary_concat_input: bool = False
    loss_ignore_index: int = 255

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="resnet",
            default_config_kwargs={
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            },
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["UperNetConfig"]
