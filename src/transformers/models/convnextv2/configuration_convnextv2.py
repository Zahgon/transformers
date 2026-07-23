
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/convnextv2-tiny-1k-224")
@strict
class ConvNextV2Config(BackboneConfigMixin, PreTrainedConfig):

    model_type = "convnextv2"

    num_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 4
    num_stages: int = 4
    hidden_sizes: list[int] | tuple[int, ...] | None = (96, 192, 384, 768)
    depths: list[int] | tuple[int, ...] | None = (3, 3, 9, 3)
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    drop_path_rate: float | int = 0.0
    image_size: int | list[int] | tuple[int, int] = 224
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["ConvNextV2Config"]
