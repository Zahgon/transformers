
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
@strict
class DINOv3ConvNextConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "dinov3_convnext"

    num_channels: int = 3
    hidden_sizes: list[int] | None = None
    depths: list[int] | None = None
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    layer_scale_init_value: float = 1e-6
    drop_path_rate: float | int = 0.0
    image_size: int | list[int] | tuple[int, int] = 224
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.hidden_sizes = [96, 192, 384, 768] if self.hidden_sizes is None else self.hidden_sizes
        self.depths = [3, 3, 9, 3] if self.depths is None else self.depths
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)

    @property
    def num_stages(self) -> int:
        pass


__all__ = ["DINOv3ConvNextConfig"]
