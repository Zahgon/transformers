
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


ZOEDEPTH_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Intel/zoedepth-nyu": "https://huggingface.co/Intel/zoedepth-nyu/resolve/main/config.json",
}


@auto_docstring(checkpoint="Intel/zoedepth-nyu")
@strict
class ZoeDepthConfig(PreTrainedConfig):

    model_type = "zoedepth"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    batch_norm_eps: float = 1e-05
    readout_type: Literal["ignore", "add", "project"] = "project"
    reassemble_factors: list[int | float] | tuple[int | float, ...] = (4, 2, 1, 0.5)
    neck_hidden_sizes: list[int] | tuple[int, ...] = (96, 192, 384, 768)
    fusion_hidden_size: int = 256
    head_in_index: int = -1
    use_batch_norm_in_fusion_residual: bool = False
    use_bias_in_fusion_residual: bool | None = None
    num_relative_features: int = 32
    add_projection: bool = False
    bottleneck_features: int = 256
    num_attractors: list[int] | tuple[int, ...] = (16, 8, 4, 1)
    bin_embedding_dim: int = 128
    attractor_alpha: int = 1000
    attractor_gamma: int = 2
    attractor_kind: Literal["mean", "sum"] = "mean"
    min_temp: float = 0.0212
    max_temp: float = 50.0
    bin_centers_type: str = "softplus"
    bin_configurations: list[dict] | None = None
    num_patch_transformer_layers: int | None = None
    patch_transformer_hidden_size: int | None = None
    patch_transformer_intermediate_size: int | None = None
    patch_transformer_num_attention_heads: int | None = None

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="beit",
            default_config_kwargs={
                "image_size": 384,
                "num_hidden_layers": 24,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "use_relative_position_bias": True,
                "reshape_hidden_states": False,
                "out_features": ["stage6", "stage12", "stage18", "stage24"],
            },
            **kwargs,
        )
        self.bin_configurations = self.bin_configurations or [{"n_bins": 64, "min_depth": 0.001, "max_depth": 10.0}]

        super().__post_init__(**kwargs)


__all__ = ["ZOEDEPTH_PRETRAINED_CONFIG_ARCHIVE_MAP", "ZoeDepthConfig"]
