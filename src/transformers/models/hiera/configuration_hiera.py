
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/hiera-base-224")
@strict
class HieraConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "hiera"

    attribute_map = {"num_hidden_layers": "num_layers"}

    embed_dim: int = 96
    image_size: list[int] | tuple[int, ...] = (224, 224)
    patch_size: list[int] | tuple[int, ...] = (7, 7)
    patch_stride: list[int] | tuple[int, ...] = (4, 4)
    patch_padding: list[int] | tuple[int, ...] = (3, 3)
    mlp_ratio: float = 4.0
    depths: list[int] | tuple[int, ...] = (2, 3, 16, 3)
    num_heads: list[int] | tuple[int, ...] = (1, 2, 4, 8)
    embed_dim_multiplier: float | int = 2.0
    num_query_pool: int = 3
    query_stride: list[int] | tuple[int, ...] = (2, 2)
    masked_unit_size: list[int] | tuple[int, ...] = (8, 8)
    masked_unit_attention: list[bool] | tuple[bool, ...] = (True, True, False, False)
    drop_path_rate: float | int = 0.0
    num_channels: int = 3
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    layer_norm_init: float = 1.0
    layer_norm_eps: float = 1e-6
    decoder_hidden_size: int | None = None
    decoder_depth: int | None = None
    decoder_num_heads: int | None = None
    normalize_pixel_loss: bool | None = True
    mask_ratio: float = 0.6
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.hidden_size = int(self.embed_dim * self.embed_dim_multiplier ** (len(self.depths) - 1))
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["HieraConfig"]
