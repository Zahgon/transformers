
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="hustvl/vitmatte-small-composition-1k")
@strict
class VitMatteConfig(PreTrainedConfig):

    model_type = "vitmatte"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 384
    batch_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    convstream_hidden_sizes: list[int] | tuple[int, ...] = (48, 96, 192)
    fusion_hidden_sizes: list[int] | tuple[int, ...] = (256, 128, 64, 32)

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="vitdet",
            default_config_kwargs={"out_features": ["stage4"]},
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["VitMatteConfig"]
