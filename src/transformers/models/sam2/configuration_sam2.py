
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="facebook/sam2.1-hiera-tiny")
@strict
class Sam2HieraDetConfig(PreTrainedConfig):

    base_config_key = "backbone_config"
    model_type = "sam2_hiera_det_model"

    hidden_size: int = 96
    num_attention_heads: int = 1
    num_channels: int = 3
    image_size: int | list[int] | None = None
    patch_kernel_size: int | list[int] | None = None
    patch_stride: int | list[int] | None = None
    patch_padding: int | list[int] | None = None
    query_stride: int | list[int] | None = None
    window_positional_embedding_background_size: list[int] | None = None
    num_query_pool_stages: int = 3
    blocks_per_stage: list[int] | None = None
    embed_dim_per_stage: list[int] | None = None
    num_attention_heads_per_stage: list[int] | None = None
    window_size_per_stage: list[int] | None = None
    global_attention_blocks: list[int] | None = None
    mlp_ratio: float = 4.0
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.image_size = self.image_size if self.image_size is not None else [1024, 1024]
        self.patch_kernel_size = self.patch_kernel_size if self.patch_kernel_size is not None else [7, 7]
        self.patch_stride = self.patch_stride if self.patch_stride is not None else [4, 4]
        self.patch_padding = self.patch_padding if self.patch_padding is not None else [3, 3]
        self.query_stride = self.query_stride if self.query_stride is not None else [2, 2]
        self.window_positional_embedding_background_size = (
            self.window_positional_embedding_background_size
            if self.window_positional_embedding_background_size is not None
            else [7, 7]
        )
        self.blocks_per_stage = self.blocks_per_stage if self.blocks_per_stage is not None else [1, 2, 7, 2]
        self.embed_dim_per_stage = (
            self.embed_dim_per_stage if self.embed_dim_per_stage is not None else [96, 192, 384, 768]
        )
        self.num_attention_heads_per_stage = (
            self.num_attention_heads_per_stage if self.num_attention_heads_per_stage is not None else [1, 2, 4, 8]
        )
        self.window_size_per_stage = (
            self.window_size_per_stage if self.window_size_per_stage is not None else [8, 4, 14, 7]
        )
        self.global_attention_blocks = (
            self.global_attention_blocks if self.global_attention_blocks is not None else [5, 7, 9]
        )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/sam2.1-hiera-tiny")
@strict
class Sam2VisionConfig(PreTrainedConfig):

    base_config_key = "vision_config"
    model_type = "sam2_vision_model"
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
            [768, 384, 192, 96] if self.backbone_channel_list is None else self.backbone_channel_list
        )
        self.backbone_feature_sizes = (
            [[256, 256], [128, 128], [64, 64]] if self.backbone_feature_sizes is None else self.backbone_feature_sizes
        )
        self.fpn_top_down_levels = [2, 3] if self.fpn_top_down_levels is None else self.fpn_top_down_levels

        if isinstance(self.backbone_config, dict):
            self.backbone_config["model_type"] = self.backbone_config.get("model_type", "sam2_hiera_det_model")
            self.backbone_config = CONFIG_MAPPING[self.backbone_config["model_type"]](**self.backbone_config)
        elif self.backbone_config is None:
            self.backbone_config = Sam2HieraDetConfig()

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/sam2.1-hiera-tiny")
@strict
class Sam2PromptEncoderConfig(PreTrainedConfig):

    base_config_key = "prompt_encoder_config"

    hidden_size: int = 256
    image_size: int | list[int] | tuple[int, int] = 1024
    patch_size: int | list[int] | tuple[int, int] = 16
    mask_input_channels: int = 16
    num_point_embeddings: int = 4
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    scale: int = 1


@auto_docstring(checkpoint="facebook/sam2.1-hiera-tiny")
@strict
class Sam2MaskDecoderConfig(PreTrainedConfig):

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


@auto_docstring(checkpoint="facebook/sam2.1-hiera-tiny")
@strict
class Sam2Config(PreTrainedConfig):

    model_type = "sam2"
    sub_configs = {
        "vision_config": AutoConfig,
        "prompt_encoder_config": Sam2PromptEncoderConfig,
        "mask_decoder_config": Sam2MaskDecoderConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "sam2_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["sam2_vision_model"]()

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = Sam2PromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = Sam2PromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = Sam2MaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = Sam2MaskDecoderConfig()

        super().__post_init__(**kwargs)


__all__ = [
    "Sam2Config",
    "Sam2HieraDetConfig",
    "Sam2VisionConfig",
    "Sam2PromptEncoderConfig",
    "Sam2MaskDecoderConfig",
]
