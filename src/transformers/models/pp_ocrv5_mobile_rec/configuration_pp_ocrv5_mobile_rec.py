

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv5_mobile_rec_safetensors")
@strict
class PPOCRV5MobileRecConfig(PreTrainedConfig):

    model_type = "pp_ocrv5_mobile_rec"
    sub_configs = {"backbone_config": AutoConfig}

    hidden_act: str = "silu"
    backbone_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 120
    mlp_ratio: float = 2.0
    depth: int = 2
    head_out_channels: int = 18385
    conv_kernel_size: list | None = None
    qkv_bias: bool = True
    num_attention_heads: int = 8
    attention_dropout: float | int = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self, **kwargs):
        if self.conv_kernel_size is None:
            self.conv_kernel_size = [1, 3]
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet_v3",
            default_config_kwargs={
                "scale": 0.75,
                "out_features": ["stage2", "stage3", "stage4", "stage5"],
                "out_indices": [2, 3, 4, 5],
                "divisor": 16,
            },
            **kwargs,
        )
        if self.conv_kernel_size is None:
            self.conv_kernel_size = [1, 3]
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="hgnet_v2",
            default_config_kwargs={
                "arch": "L",
                "return_idx": [0, 1, 2, 3],
                "freeze_stem_only": True,
                "freeze_at": 0,
                "freeze_norm": True,
                "lr_mult_list": [1.0, 1.0, 1.0, 1.0, 1.0],
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
                "stage_downsample": [True, True, True, True],
                "stem_strides": [2, 1, 1, 1, 1],
                "stage_downsample_strides": [[2, 1], [1, 2], [2, 1], [2, 1]],
            },
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["PPOCRV5MobileRecConfig"]
