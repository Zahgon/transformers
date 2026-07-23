from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/maskformer-swin-base-ade")
@strict
class MaskFormerDetrConfig(PreTrainedConfig):

    model_type = "detr"
    sub_configs = {"backbone_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }

    backbone_config: dict | PreTrainedConfig | None = None
    num_channels: int = 3
    num_queries: int = 100
    encoder_layers: int = 6
    encoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 8
    decoder_layers: int = 6
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 8
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    is_encoder_decoder: bool = True
    activation_function: str = "relu"
    d_model: int = 256
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    init_xavier_std: float = 1.0
    auxiliary_loss: bool = False
    position_embedding_type: str = "sine"
    dilation: bool = False
    class_cost: int = 1
    bbox_cost: int = 5
    giou_cost: int = 2
    mask_loss_coefficient: int = 1
    dice_loss_coefficient: int = 1
    bbox_loss_coefficient: int = 5
    giou_loss_coefficient: int = 2
    eos_coefficient: float = 0.1

    def __post_init__(self, **kwargs):
        backbone_kwargs = kwargs.get("backbone_kwargs", {})
        timm_default_kwargs = {
            "num_channels": backbone_kwargs.get("num_channels", self.num_channels),
            "features_only": True,
            "use_pretrained_backbone": False,
            "out_indices": backbone_kwargs.get("out_indices", [1, 2, 3, 4]),
        }
        if self.dilation:
            timm_default_kwargs["output_stride"] = backbone_kwargs.get("output_stride", 16)

        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_backbone="resnet50",
            default_config_type="resnet",
            default_config_kwargs={"out_features": ["stage4"]},
            timm_default_kwargs=timm_default_kwargs,
            **kwargs,
        )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/maskformer-swin-base-ade")
@strict
class MaskFormerConfig(PreTrainedConfig):

    model_type = "maskformer"
    sub_configs = {"backbone_config": AutoConfig, "decoder_config": AutoConfig}
    attribute_map = {"hidden_size": "mask_feature_size"}
    backbones_supported = ["resnet", "swin"]
    decoders_supported = ["detr"]

    fpn_feature_size: int = 256
    mask_feature_size: int = 256
    no_object_weight: float = 0.1
    use_auxiliary_loss: bool = False
    backbone_config: dict | PreTrainedConfig | None = None
    decoder_config: dict | PreTrainedConfig | None = None
    init_std: float = 0.02
    init_xavier_std: float = 1.0
    dice_weight: float = 1.0
    cross_entropy_weight: float = 1.0
    mask_weight: float = 20.0
    output_auxiliary_logits: bool | None = None

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="swin",
            default_config_kwargs={
                "depths": [2, 2, 18, 2],
                "drop_path_rate": 0.3,
                "image_size": 384,
                "embed_dim": 128,
                "num_heads": [4, 8, 16, 32],
                "window_size": 12,
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            },
            **kwargs,
        )

        if self.backbone_config is not None and self.backbone_config.model_type not in self.backbones_supported:
            logger.warning_once(
                f"Backbone {self.backbone_config.model_type} is not a supported model and may not be compatible with MaskFormer. "
                f"Supported model types: {','.join(self.backbones_supported)}"
            )

        if self.decoder_config is None:
            self.decoder_config = MaskFormerDetrConfig()
        else:
            decoder_type = (
                self.decoder_config.pop("model_type")
                if isinstance(self.decoder_config, dict)
                else self.decoder_config.model_type
            )
            if decoder_type not in self.decoders_supported:
                raise ValueError(
                    f"Transformer Decoder {decoder_type} not supported, please use one of"
                    f" {','.join(self.decoders_supported)}"
                )
            if isinstance(self.decoder_config, dict):
                config_class = CONFIG_MAPPING[decoder_type]
                self.decoder_config = config_class.from_dict(self.decoder_config)

        self.num_attention_heads = self.decoder_config.encoder_attention_heads
        self.num_hidden_layers = self.decoder_config.num_hidden_layers
        super().__post_init__(**kwargs)


__all__ = ["MaskFormerConfig", "MaskFormerDetrConfig"]
