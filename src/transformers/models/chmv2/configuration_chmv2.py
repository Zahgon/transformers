
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="facebook/dinov3-vitl16-chmv2-dpt-head")
@strict
class CHMv2Config(PreTrainedConfig):

    model_type = "chmv2"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    patch_size: int = 16
    initializer_range: float = 0.02
    reassemble_factors: list[float | int] | None = None
    post_process_channels: list[int] | None = None
    fusion_hidden_size: int = 256
    head_hidden_size: int = 128
    number_output_channels: int = 256
    readout_type: str = "project"
    min_depth: float = 0.001
    max_depth: float = 96.0
    bins_strategy: Literal["linear", "log", "chmv2_mixlog"] = "chmv2_mixlog"
    norm_strategy: Literal["linear", "softmax", "sigmoid", "chmv2_mixlog"] = "chmv2_mixlog"

    def __post_init__(self, **kwargs):
        if self.reassemble_factors is None:
            self.reassemble_factors = [4, 2, 1, 0.5]
        if self.post_process_channels is None:
            self.post_process_channels = [128, 256, 512, 1024]

        default_config_kwargs = {
            "image_size": 416,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "num_register_tokens": 4,
            "key_bias": True,
            "out_indices": [6, 12, 18, 24],
            "reshape_hidden_states": True,
            "apply_layernorm": True,
            "layer_norm_eps": 1e-6,
            "return_class_token": True,
        }

        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="dinov3_vit",
            default_config_kwargs=default_config_kwargs,
            **kwargs,
        )

        super().__post_init__(**kwargs)


__all__ = ["CHMv2Config"]
