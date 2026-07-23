
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/resnet-50")
@strict
class RTDetrResNetConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "rt_detr_resnet"
    layer_types = ["basic", "bottleneck"]

    num_channels: int = 3
    embedding_size: int = 64
    hidden_sizes: list[int] | tuple[int, ...] = (256, 512, 1024, 2048)
    depths: list[int] | tuple[int, ...] = (3, 4, 6, 3)
    layer_type: str = "bottleneck"
    hidden_act: str = "relu"
    downsample_in_first_stage: bool = False
    downsample_in_bottleneck: bool = False
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        self.hidden_sizes = list(self.hidden_sizes)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["RTDetrResNetConfig"]
