
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/focalnet-tiny")
@strict
class FocalNetConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "focalnet"

    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 4
    num_channels: int = 3
    embed_dim: int = 96
    use_conv_embed: bool = False
    hidden_sizes: list[int] | tuple[int, ...] = (192, 384, 768, 768)
    depths: list[int] | tuple[int, ...] = (2, 2, 6, 2)
    focal_levels: list[int] | tuple[int, ...] = (2, 2, 2, 2)
    focal_windows: list[int] | tuple[int, ...] = (3, 3, 3, 3)
    hidden_act: str = "gelu"
    mlp_ratio: float = 4.0
    hidden_dropout_prob: float | int = 0.0
    drop_path_rate: float | int = 0.1
    use_layerscale: bool = False
    layerscale_value: float = 1e-4
    use_post_layernorm: bool = False
    use_post_layernorm_in_modulation: bool = False
    normalize_modulator: bool = False
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    encoder_stride: int = 32
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["FocalNetConfig"]
