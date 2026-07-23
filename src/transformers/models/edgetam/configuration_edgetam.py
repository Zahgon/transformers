from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="yonigozlan/EdgeTAM-hf")
@strict
class EdgeTamVisionConfig(PreTrainedConfig):

    base_config_key = "vision_config"
    model_type = "edgetam_vision_model"
    sub_configs = {
        "backbone_config": AutoConfig,
    }

    backbone_config: dict | PreTrainedConfig | None = None
    backbone_channel_list: list[int] | None = None
    backbone_feature_sizes: list | None = None
    fpn_hidden_size: int = 256
    fpn_kernel_size: int = 1
    fpn_stride: int = 1
    fpn_padding: int = 0
    fpn_top_down_levels: list[int] | None = None
    num_feature_levels: int = 3
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.backbone_channel_list = (
            [384, 192, 96, 48] if self.backbone_channel_list is None else self.backbone_channel_list
        )
        self.backbone_feature_sizes = (
            [[256, 256], [128, 128], [64, 64]] if self.backbone_feature_sizes is None else self.backbone_feature_sizes
        )
        self.fpn_top_down_levels = [2, 3] if self.fpn_top_down_levels is None else self.fpn_top_down_levels

        if isinstance(self.backbone_config, dict):
            self.backbone_config["model_type"] = self.backbone_config.get("model_type", "timm_wrapper")
            self.backbone_config = CONFIG_MAPPING[self.backbone_config["model_type"]](**self.backbone_config)
        elif self.backbone_config is None:
            self.backbone_config = AutoConfig.from_pretrained(
                "timm/repvit_m1.dist_in1k",
                model_args={"in_chans": 3, "features_only": True, "out_indices": [0, 1, 2, 3]},
            )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="yonigozlan/EdgeTAM-hf")
@strict
class EdgeTamPromptEncoderConfig(PreTrainedConfig):

    base_config_key = "prompt_encoder_config"

    hidden_size: int = 256
    image_size: int | list[int] | tuple[int, int] = 1024
    patch_size: int | list[int] | tuple[int, int] = 16
    mask_input_channels: int = 16
    num_point_embeddings: int = 4
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    scale: int = 1


@auto_docstring(checkpoint="yonigozlan/EdgeTAM-hf")
@strict
class EdgeTamMaskDecoderConfig(PreTrainedConfig):

    base_config_key = "mask_decoder_config"

    hidden_size: int = 256
    hidden_act: str = "gelu"
    mlp_dim: int = 2048
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    attention_downsample_rate: int = 2
    num_multimask_outputs: int = 3
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 256
    dynamic_multimask_via_stability: bool = True
    dynamic_multimask_stability_delta: float = 0.05
    dynamic_multimask_stability_thresh: float = 0.98


@auto_docstring(checkpoint="yonigozlan/EdgeTAM-hf")
@strict
class EdgeTamConfig(PreTrainedConfig):

    model_type = "edgetam"
    sub_configs = {
        "vision_config": AutoConfig,
        "prompt_encoder_config": EdgeTamPromptEncoderConfig,
        "mask_decoder_config": EdgeTamMaskDecoderConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "edgetam_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["edgetam_vision_model"]()

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = EdgeTamPromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = EdgeTamPromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = EdgeTamMaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = EdgeTamMaskDecoderConfig()

        super().__post_init__(**kwargs)


__all__ = ["EdgeTamConfig", "EdgeTamVisionConfig", "EdgeTamPromptEncoderConfig", "EdgeTamMaskDecoderConfig"]
