

from collections.abc import Sequence

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="ustc-community/dfine_x_coco")
@strict
class HGNetV2Config(BackboneConfigMixin, PreTrainedConfig):

    model_type = "hgnet_v2"

    num_channels: int = 3
    embedding_size: int = 64
    depths: list[int] | tuple[int, ...] = (3, 4, 6, 3)
    hidden_sizes: list[int] | tuple[int, ...] = (256, 512, 1024, 2048)
    hidden_act: str = "relu"
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    stem_channels: list[int] | tuple[int, ...] = (3, 32, 48)
    stem_strides: Sequence[int | list[int] | tuple[int, ...]] = (2, 1, 1, 2, 1)
    stage_in_channels: list[int] | tuple[int, ...] = (48, 128, 512, 1024)
    stage_mid_channels: list[int] | tuple[int, ...] = (48, 96, 192, 384)
    stage_out_channels: list[int] | tuple[int, ...] = (128, 512, 1024, 2048)
    stage_num_blocks: list[int] | tuple[int, ...] = (1, 1, 3, 1)
    stage_downsample: list[bool] | tuple[bool, ...] = (False, True, True, True)
    stage_downsample_strides: Sequence[int | list[int] | tuple[int, ...]] = (2, 2, 2, 2)
    stage_light_block: list[bool] | tuple[bool, ...] = (False, False, True, True)
    stage_kernel_size: list[int] | tuple[int, ...] = (3, 3, 5, 5)
    stage_numb_of_layers: list[int] | tuple[int, ...] = (6, 6, 6, 6)
    use_learnable_affine_block: bool = False
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        self.hidden_sizes = list(self.hidden_sizes)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["HGNetV2Config"]
