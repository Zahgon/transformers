

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3TrackerPromptEncoderConfig(PreTrainedConfig):

    base_config_key = "prompt_encoder_config"

    hidden_size: int = 256

    image_size: int | list[int] | tuple[int, int] = 1008
    patch_size: int | list[int] | tuple[int, int] = 14
    mask_input_channels: int = 16
    num_point_embeddings: int = 4
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    scale: int = 1


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3TrackerMaskDecoderConfig(PreTrainedConfig):

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


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3TrackerConfig(PreTrainedConfig):

    model_type = "sam3_tracker"
    sub_configs = {
        "vision_config": AutoConfig,
        "prompt_encoder_config": Sam3TrackerPromptEncoderConfig,
        "mask_decoder_config": Sam3TrackerMaskDecoderConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "sam3_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["sam3_vision_model"](
                backbone_feature_sizes=[[288, 288], [144, 144], [72, 72]]
            )

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = Sam3TrackerPromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = Sam3TrackerPromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = Sam3TrackerMaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = Sam3TrackerMaskDecoderConfig()

        super().__post_init__(**kwargs)


__all__ = ["Sam3TrackerConfig", "Sam3TrackerPromptEncoderConfig", "Sam3TrackerMaskDecoderConfig"]
