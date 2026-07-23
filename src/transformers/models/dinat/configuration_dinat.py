
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="shi-labs/dinat-mini-in1k-224")
@strict
class DinatConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "dinat"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    patch_size: int | list[int] | tuple[int, int] = 4
    num_channels: int = 3
    embed_dim: int = 64
    depths: list[int] | tuple[int, ...] = (3, 4, 6, 5)
    num_heads: list[int] | tuple[int, ...] = (2, 4, 8, 16)
    kernel_size: int = 7
    dilations: list | tuple | None = None
    mlp_ratio: float = 3.0
    qkv_bias: bool = True
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    drop_path_rate: float | int = 0.1
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    layer_scale_init_value: float = 0.0
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.num_layers = len(self.depths)
        self.dilations = self.dilations or [[1, 8, 1], [1, 4, 1, 4], [1, 2, 1, 2, 1, 2], [1, 1, 1, 1, 1]]

        self.hidden_size = int(self.embed_dim * 2 ** (len(self.depths) - 1))
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["DinatConfig"]
