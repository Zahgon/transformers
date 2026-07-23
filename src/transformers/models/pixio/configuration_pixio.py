from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/pixio-huge")
@strict
class PixioConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "pixio"

    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    mlp_ratio: int = 4
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int | list[int] | tuple[int, int] = 256
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    drop_path_rate: float | int = 0.0
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    apply_layernorm: bool = True
    reshape_hidden_states: bool = True
    n_cls_tokens: int = 8

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["PixioConfig"]
