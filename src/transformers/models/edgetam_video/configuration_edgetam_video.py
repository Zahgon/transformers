
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="yonigozlan/EdgeTAM-hf")
@strict
class EdgeTamVideoPromptEncoderConfig(PreTrainedConfig):

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
class EdgeTamVideoMaskDecoderConfig(PreTrainedConfig):

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
class EdgeTamVideoConfig(PreTrainedConfig):

    model_type = "edgetam_video"
    sub_configs = {
        "vision_config": AutoConfig,
        "prompt_encoder_config": EdgeTamVideoPromptEncoderConfig,
        "mask_decoder_config": EdgeTamVideoMaskDecoderConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    num_maskmem: int = 7
    image_size: int | list[int] | tuple[int, int] = 1024
    sigmoid_scale_for_mem_enc: float = 20.0
    sigmoid_bias_for_mem_enc: float = -10.0
    enable_occlusion_spatial_embedding: bool = True
    multimask_output_in_sam: bool = True
    multimask_min_pt_num: int = 0
    multimask_max_pt_num: int = 1
    multimask_output_for_tracking: bool = True
    max_object_pointers_in_encoder: int = 16
    max_cond_frame_num: int = -1
    enable_temporal_pos_encoding_for_object_pointers: bool = True

    memory_attention_hidden_size: int = 256
    memory_attention_num_layers: int = 2
    memory_attention_num_attention_heads: int = 1
    memory_attention_downsample_rate: int = 1
    memory_attention_mlp_hidden_size: int = 2048
    memory_attention_mlp_hidden_act: str = "relu"
    memory_attention_dropout: float | int = 0.1
    memory_attention_rope_theta: float | int = 10000
    memory_attention_rope_feat_sizes: list | None = None
    memory_attention_rope_k_sizes: list | None = None
    memory_attention_rope_dropout: float | int = 0.1

    perceiver_resampler_num_latents: int = 256
    perceiver_resampler_num_latents_2d: int = 256
    perceiver_resampler_hidden_size: int = 64
    perceiver_resampler_mlp_intermediate_size: int = 256
    perceiver_resampler_num_attention_heads: int = 1
    perceiver_resampler_attention_head_dim: int = 64
    perceiver_resampler_num_layers: int = 2
    perceiver_resampler_hidden_dropout: float | int = 0.0
    perceiver_resampler_attention_dropout: float | int = 0.0

    memory_encoder_hidden_size: int = 256
    memory_encoder_output_channels: int = 64
    mask_downsampler_embed_dim: int = 256
    memory_fuser_intermediate_dim: int = 1024
    mask_downsampler_kernel_size: int = 3
    mask_downsampler_stride: int = 2
    mask_downsampler_padding: int = 1
    mask_downsampler_total_stride: int = 16
    mask_downsampler_hidden_act: str = "gelu"
    memory_fuser_num_layers: int = 2
    memory_fuser_embed_dim: int = 256
    memory_fuser_kernel_size: int = 7
    memory_fuser_padding: int = 3
    memory_fuser_layer_scale_init_value: float = 1e-6
    memory_fuser_hidden_act: str = "gelu"

    def __post_init__(self, **kwargs):
        self.prompt_encoder_config = self.prompt_encoder_config if self.prompt_encoder_config is not None else {}
        self.mask_decoder_config = self.mask_decoder_config if self.mask_decoder_config is not None else {}
        self.memory_attention_rope_feat_sizes = (
            [64, 64] if self.memory_attention_rope_feat_sizes is None else self.memory_attention_rope_feat_sizes
        )
        self.memory_attention_rope_k_sizes = (
            [16, 16] if self.memory_attention_rope_k_sizes is None else self.memory_attention_rope_k_sizes
        )

        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "sam2_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["sam2_vision_model"]()

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = EdgeTamVideoPromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = EdgeTamVideoPromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = EdgeTamVideoMaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = EdgeTamVideoMaskDecoderConfig()
        super().__post_init__(**kwargs)


__all__ = ["EdgeTamVideoMaskDecoderConfig", "EdgeTamVideoPromptEncoderConfig", "EdgeTamVideoConfig"]
