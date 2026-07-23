

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="facebook/sam3")
@strict
class Sam3TrackerVideoPromptEncoderConfig(PreTrainedConfig):

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
class Sam3TrackerVideoMaskDecoderConfig(PreTrainedConfig):

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
class Sam3TrackerVideoConfig(PreTrainedConfig):

    model_type = "sam3_tracker_video"
    sub_configs = {
        "vision_config": AutoConfig,
        "prompt_encoder_config": Sam3TrackerVideoPromptEncoderConfig,
        "mask_decoder_config": Sam3TrackerVideoMaskDecoderConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    num_maskmem: int = 7
    sigmoid_scale_for_mem_enc: float = 20.0
    sigmoid_bias_for_mem_enc: float = -10.0
    enable_occlusion_spatial_embedding: bool = True
    multimask_output_in_sam: bool = True
    multimask_min_pt_num: int = 0
    multimask_max_pt_num: int = 1
    multimask_output_for_tracking: bool = True
    max_object_pointers_in_encoder: int = 16
    max_cond_frame_num: int = 4
    enable_temporal_pos_encoding_for_object_pointers: bool = True
    memory_attention_hidden_size: int = 256
    memory_attention_num_layers: int = 4
    memory_attention_num_attention_heads: int = 1
    memory_attention_downsample_rate: int = 1
    memory_attention_feed_forward_hidden_size: int = 2048
    memory_attention_feed_forward_hidden_act: str = "relu"
    memory_attention_dropout: float | int = 0.1
    memory_attention_rope_theta: int = 10000
    memory_attention_rope_feat_sizes: list | None = None
    memory_attention_rope_dropout: float | int = 0.1
    memory_encoder_hidden_size: int = 256
    memory_encoder_output_channels: int = 64
    mask_downsampler_embed_dim: int = 256
    mask_downsampler_kernel_size: int = 3
    mask_downsampler_stride: int = 2
    mask_downsampler_padding: int = 1
    mask_downsampler_total_stride: int = 16
    mask_downsampler_hidden_act: str = "gelu"
    memory_fuser_num_layers: int = 2
    memory_fuser_embed_dim: int = 256
    memory_fuser_intermediate_dim: int = 1024
    memory_fuser_kernel_size: int = 7
    memory_fuser_padding: int = 3
    memory_fuser_layer_scale_init_value: float = 1e-6
    memory_fuser_hidden_act: str = "gelu"

    def __post_init__(self, **kwargs):
        self.memory_attention_rope_feat_sizes = (
            [72, 72] if self.memory_attention_rope_feat_sizes is None else self.memory_attention_rope_feat_sizes
        )

        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "sam3_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["sam3_vision_model"](
                backbone_feature_sizes=[[288, 288], [144, 144], [72, 72]]
            )

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = Sam3TrackerVideoPromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = Sam3TrackerVideoPromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = Sam3TrackerVideoMaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = Sam3TrackerVideoMaskDecoderConfig()

        self.image_size = kwargs.pop("image_size", 1008)
        super().__post_init__(**kwargs)

    @property
    def image_size(self):
        pass

    @image_size.setter
    def image_size(self, value):
        pass


__all__ = ["Sam3TrackerVideoMaskDecoderConfig", "Sam3TrackerVideoPromptEncoderConfig", "Sam3TrackerVideoConfig"]
