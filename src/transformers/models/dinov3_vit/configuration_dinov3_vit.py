
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/dinov3-vits16-pretrain-lvd1689m")
@strict
class DINOv3ViTConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "dinov3_vit"

    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    hidden_act: str = "gelu"
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    rope_theta: float = 100.0
    image_size: int | list[int] | tuple[int, int] = 224
    num_channels: int = 3
    query_bias: bool = True
    key_bias: bool = False
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float | int = 0.0
    use_gated_mlp: bool = False
    num_register_tokens: int = 0
    pos_embed_shift: float | None = None
    pos_embed_jitter: float | None = None
    pos_embed_rescale: float | None = 2.0
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    apply_layernorm: bool = True
    reshape_hidden_states: bool = True

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{i}" for i in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["DINOv3ViTConfig"]
