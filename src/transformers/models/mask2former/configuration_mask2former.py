
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/mask2former-swin-small-coco-instance")
@strict
class Mask2FormerConfig(PreTrainedConfig):

    model_type = "mask2former"
    sub_configs = {"backbone_config": AutoConfig}
    backbones_supported = ["swin"]
    attribute_map = {"hidden_size": "hidden_dim", "num_hidden_layers": "decoder_layers"}

    backbone_config: dict | PreTrainedConfig | None = None
    feature_size: int = 256
    mask_feature_size: int = 256
    hidden_dim: int = 256
    encoder_feedforward_dim: int = 1024
    activation_function: str = "relu"
    encoder_layers: int = 6
    decoder_layers: int = 10
    num_attention_heads: int = 8
    dropout: float | int = 0.0
    dim_feedforward: int = 2048
    pre_norm: bool = False
    enforce_input_projection: bool = False
    common_stride: int = 4
    ignore_value: int = 255
    num_queries: int = 100
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    train_num_points: int = 12544
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    init_std: float = 0.02
    init_xavier_std: float = 1.0
    use_auxiliary_loss: bool = True
    feature_strides: list[int] | tuple[int, ...] = (4, 8, 16, 32)
    output_auxiliary_logits: bool | None = None

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="swin",
            default_config_kwargs={
                "depths": [2, 2, 18, 2],
                "drop_path_rate": 0.3,
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            },
            **kwargs,
        )

        if self.backbone_config.model_type not in self.backbones_supported:
            logger.warning_once(
                f"Backbone {self.backbone_config.model_type} is not a supported model and may not be compatible with Mask2Former. "
                f"Supported model types: {','.join(self.backbones_supported)}"
            )

        super().__post_init__(**kwargs)


__all__ = ["Mask2FormerConfig"]
