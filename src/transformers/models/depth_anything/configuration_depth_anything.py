
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="LiheYoung/depth-anything-small-hf")
@strict
class DepthAnythingConfig(PreTrainedConfig):

    model_type = "depth_anything"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    patch_size: int | list[int] | tuple[int, int] = 14
    initializer_range: float = 0.02
    reassemble_hidden_size: int = 384
    reassemble_factors: list[int | float] | tuple[int | float, ...] = (4, 2, 1, 0.5)
    neck_hidden_sizes: list[int] | tuple[int, ...] = (48, 96, 192, 384)
    fusion_hidden_size: int = 64
    head_in_index: int = -1
    head_hidden_size: int = 32
    depth_estimation_type: str = "relative"
    max_depth: int | None = None

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="dinov2",
            default_config_kwargs={
                "image_size": 518,
                "hidden_size": 384,
                "num_attention_heads": 6,
                "out_indices": [9, 10, 11, 12],
                "reshape_hidden_states": False,
            },
            **kwargs,
        )

        self.max_depth = self.max_depth if self.max_depth else 1
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["DepthAnythingConfig"]
