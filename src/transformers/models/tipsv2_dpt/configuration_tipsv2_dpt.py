
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="google/tipsv2-b14-dpt")
@strict
class Tipsv2DptConfig(PreTrainedConfig):

    model_type = "tipsv2_dpt"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    neck_hidden_sizes: list[int] | tuple[int, ...] | None = None
    fusion_hidden_size: int = 256
    reassemble_factors: list[int | float] | tuple[int | float, ...] | None = None
    readout_activation: str = "gelu_pytorch_tanh"
    num_depth_bins: int = 256
    min_depth: float = 0.001
    max_depth: float = 10.0
    depth_decoder_activation: str = "relu"
    semantic_loss_ignore_index: int = 255

    def __post_init__(self, **kwargs):
        if self.neck_hidden_sizes is None:
            self.neck_hidden_sizes = [96, 192, 384, 768]
        if self.reassemble_factors is None:
            self.reassemble_factors = [4, 2, 1, 0.5]

        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="tipsv2_vision_model",
            default_config_kwargs={
                "out_indices": [3, 6, 9, 12],
                "apply_layernorm": True,
                "reshape_hidden_states": False,
            },
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["Tipsv2DptConfig"]
