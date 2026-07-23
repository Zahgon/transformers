

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="")
@strict
class TimmBackboneConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "timm_backbone"

    backbone: str | None = None
    num_channels: int = 3
    features_only: bool = True
    _out_indices: list[int] | None = None
    freeze_batch_norm_2d: bool = False
    output_stride: int | None = None

    def __post_init__(self, **kwargs):
        self.out_indices = self.out_indices if self.out_indices is not None else [-1]
        super().__post_init__(**kwargs)

    @property
    def out_indices(self):
        pass

    @out_indices.setter
    def out_indices(self, out_indices: tuple[int, ...] | list[int]):
        pass

    @property
    def out_features(self):
        pass

    @out_features.setter
    def out_features(self, out_features: list[str]):
        pass


__all__ = ["TimmBackboneConfig"]
