

from collections.abc import Sequence

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin, consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="PaddlePaddle/UVDoc_safetensors")
@strict
class UVDocBackboneConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "uvdoc_backbone"

    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    resnet_head: Sequence[list[int] | tuple[int, ...]] = (
        (3, 32),
        (32, 32),
    )

    resnet_configs: Sequence[Sequence[tuple[int, int, int, bool] | list[int | bool]]] = (
        (
            (32, 32, 1, False),
            (32, 32, 3, False),
            (32, 32, 3, False),
        ),
        (
            (32, 64, 1, True),
            (64, 64, 3, False),
            (64, 64, 3, False),
            (64, 64, 3, False),
        ),
        (
            (64, 128, 1, True),
            (128, 128, 3, False),
            (128, 128, 3, False),
            (128, 128, 3, False),
            (128, 128, 3, False),
            (128, 128, 3, False),
        ),
    )

    stage_configs: Sequence[Sequence[tuple[int, ...] | list[int]]] = (
        ((128, 1),),
        ((128, 2),),
        ((128, 5),),
        (
            (128, 8),
            (128, 3),
            (128, 2),
        ),
        (
            (128, 12),
            (128, 7),
            (128, 4),
        ),
        (
            (128, 18),
            (128, 12),
            (128, 6),
        ),
    )

    kernel_size: int = 5

    def __post_init__(self, **kwargs):
        self.depths = [len(stages) for stages in self.stage_configs]
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.stage_configs) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="PaddlePaddle/UVDoc_safetensors")
@strict
class UVDocConfig(PreTrainedConfig):

    model_type = "uvdoc"
    sub_configs = {"backbone_config": AutoConfig}
    backbone_config: dict | PreTrainedConfig | None = None

    hidden_act: str = "prelu"
    padding_mode: str = "reflect"
    kernel_size: int = 5
    bridge_connector: list[int] | tuple[int, ...] = (128, 128)
    out_point_positions2D: Sequence[list[int] | tuple[int, ...]] = ((128, 32), (32, 2))

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="uvdoc_backbone",
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["UVDocBackboneConfig", "UVDocConfig"]
