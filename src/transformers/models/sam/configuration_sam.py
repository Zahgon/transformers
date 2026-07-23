
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/sam-vit-huge")
@strict
class SamPromptEncoderConfig(PreTrainedConfig):

    base_config_key = "prompt_encoder_config"

    hidden_size: int = 256
    image_size: int | list[int] | tuple[int, int] = 1024
    patch_size: int | list[int] | tuple[int, int] = 16
    mask_input_channels: int = 16
    num_point_embeddings: int = 4
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6

    def __post_init__(self, **kwargs):
        self.image_embedding_size = self.image_size // self.patch_size
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/sam-vit-huge")
@strict
class SamMaskDecoderConfig(PreTrainedConfig):

    base_config_key = "mask_decoder_config"

    hidden_size: int = 256
    hidden_act: str = "relu"
    mlp_dim: int = 2048
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    attention_downsample_rate: int = 2
    num_multimask_outputs: int = 3
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 256
    layer_norm_eps: float = 1e-6


@auto_docstring(checkpoint="facebook/sam-vit-huge")
@strict
class SamVisionConfig(PreTrainedConfig):

    base_config_key = "vision_config"
    model_type = "sam_vision_model"

    hidden_size: int = 768
    output_channels: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 1024
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-06
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10
    qkv_bias: bool = True
    mlp_ratio: float = 4.0
    use_abs_pos: bool = True
    use_rel_pos: bool = True
    window_size: int = 14
    global_attn_indexes: list[int] | tuple[int, ...] = (2, 5, 8, 11)
    num_pos_feats: int = 128
    mlp_dim: int | None = None

    def __post_init__(self, **kwargs):
        self.mlp_dim = int(self.hidden_size * self.mlp_ratio) if self.mlp_dim is None else self.mlp_dim
        self.scale = self.hidden_size // 2
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/sam-vit-huge")
@strict
class SamConfig(PreTrainedConfig):

    model_type = "sam"
    sub_configs = {
        "prompt_encoder_config": SamPromptEncoderConfig,
        "mask_decoder_config": SamMaskDecoderConfig,
        "vision_config": SamVisionConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = SamVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = SamVisionConfig()

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = SamPromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = SamPromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = SamMaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = SamMaskDecoderConfig()

        super().__post_init__(**kwargs)


__all__ = ["SamConfig", "SamMaskDecoderConfig", "SamPromptEncoderConfig", "SamVisionConfig"]
