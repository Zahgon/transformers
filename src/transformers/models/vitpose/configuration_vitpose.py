
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="usyd-community/vitpose-base-simple")
@strict
class VitPoseConfig(PreTrainedConfig):

    model_type = "vitpose"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    scale_factor: int = 4
    use_simple_decoder: bool = True

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="vitpose_backbone",
            default_config_kwargs={"out_indices": [4]},
            **kwargs,
        )

        super().__post_init__(**kwargs)


__all__ = ["VitPoseConfig"]
