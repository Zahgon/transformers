
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="OpenGVLab/pvt_v2_b0")
@strict
class PvtV2Config(BackboneConfigMixin, PreTrainedConfig):

    model_type = "pvt_v2"

    image_size: int | list[int] | tuple[int, int] | dict = 224
    num_channels: int = 3
    num_encoder_blocks: int = 4
    depths: list[int] | tuple[int, ...] = (2, 2, 2, 2)
    sr_ratios: list[int] | tuple[int, ...] = (8, 4, 2, 1)
    hidden_sizes: list[int] | tuple[int, ...] = (32, 64, 160, 256)
    patch_sizes: list[int] | tuple[int, ...] = (7, 3, 3, 3)
    strides: list[int] | tuple[int, ...] = (4, 2, 2, 2)
    num_attention_heads: list[int] | tuple[int, ...] = (1, 2, 5, 8)
    mlp_ratios: list[int] | tuple[int, ...] = (8, 8, 4, 4)
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    drop_path_rate: float | int = 0.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    linear_attention: bool = False
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.image_size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else self.image_size
        self.stage_names = [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["PvtV2Config"]
