
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="yonigozlan/sam3-litetext-s0")
@strict
class Sam3LiteTextGeometryEncoderConfig(PreTrainedConfig):

    model_type = "sam3_lite_text_geometry_encoder"

    hidden_size: int = 256
    num_layers: int = 3
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float | int = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float | int = 0.0
    layer_norm_eps: float = 1e-6
    roi_size: int = 7
    initializer_range: float = 0.02


@auto_docstring(checkpoint="yonigozlan/sam3-litetext-s0")
@strict
class Sam3LiteTextDETREncoderConfig(PreTrainedConfig):

    model_type = "sam3_lite_text_detr_encoder"

    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float | int = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float | int = 0.0
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02


@auto_docstring(checkpoint="yonigozlan/sam3-litetext-s0")
@strict
class Sam3LiteTextDETRDecoderConfig(PreTrainedConfig):

    model_type = "sam3_lite_text_detr_decoder"

    hidden_size: int = 256
    num_layers: int = 6
    num_queries: int = 200
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float | int = 0.1
    hidden_act: str = "relu"
    hidden_dropout: float | int = 0.0
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02


@auto_docstring(checkpoint="yonigozlan/sam3-litetext-s0")
@strict
class Sam3LiteTextMaskDecoderConfig(PreTrainedConfig):

    model_type = "sam3_lite_text_mask_decoder"

    hidden_size: int = 256
    num_upsampling_stages: int = 3
    layer_norm_eps: float = 1e-6
    dropout: float | int = 0.0
    num_attention_heads: int = 8
    initializer_range: float = 0.02


@auto_docstring(checkpoint="yonigozlan/sam3-litetext-s0")
@strict
class Sam3LiteTextTextConfig(PreTrainedConfig):

    model_type = "sam3_lite_text_text_model"

    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    use_repmixer_blocks: bool = True
    layer_scale_init_value: float = 1e-5
    repmixer_kernel_size: int = 11


@auto_docstring(checkpoint="yonigozlan/sam3-litetext-s0")
@strict
class Sam3LiteTextConfig(PreTrainedConfig):

    model_type = "sam3_lite_text"
    sub_configs = {
        "vision_config": AutoConfig,
        "text_config": Sam3LiteTextTextConfig,
        "geometry_encoder_config": Sam3LiteTextGeometryEncoderConfig,
        "detr_encoder_config": Sam3LiteTextDETREncoderConfig,
        "detr_decoder_config": Sam3LiteTextDETRDecoderConfig,
        "mask_decoder_config": Sam3LiteTextMaskDecoderConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    geometry_encoder_config: dict | PreTrainedConfig | None = None
    detr_encoder_config: dict | PreTrainedConfig | None = None
    detr_decoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "sam3_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["sam3_vision_model"]()

        if self.text_config is None:
            self.text_config = Sam3LiteTextTextConfig()
        if isinstance(self.text_config, dict):
            self.text_config = Sam3LiteTextTextConfig(**self.text_config)

        if self.geometry_encoder_config is None:
            self.geometry_encoder_config = Sam3LiteTextGeometryEncoderConfig()
        if isinstance(self.geometry_encoder_config, dict):
            self.geometry_encoder_config = Sam3LiteTextGeometryEncoderConfig(**self.geometry_encoder_config)

        if self.detr_encoder_config is None:
            self.detr_encoder_config = Sam3LiteTextDETREncoderConfig()
        if isinstance(self.detr_encoder_config, dict):
            self.detr_encoder_config = Sam3LiteTextDETREncoderConfig(**self.detr_encoder_config)

        if self.detr_decoder_config is None:
            self.detr_decoder_config = Sam3LiteTextDETRDecoderConfig()
        if isinstance(self.detr_decoder_config, dict):
            self.detr_decoder_config = Sam3LiteTextDETRDecoderConfig(**self.detr_decoder_config)

        if self.mask_decoder_config is None:
            self.mask_decoder_config = Sam3LiteTextMaskDecoderConfig()
        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = Sam3LiteTextMaskDecoderConfig(**self.mask_decoder_config)

        super().__post_init__(**kwargs)

    @property
    def image_size(self):
        pass

    @image_size.setter
    def image_size(self, value):
        pass


__all__ = [
    "Sam3LiteTextConfig",
    "Sam3LiteTextTextConfig",
    "Sam3LiteTextGeometryEncoderConfig",
    "Sam3LiteTextDETREncoderConfig",
    "Sam3LiteTextDETRDecoderConfig",
    "Sam3LiteTextMaskDecoderConfig",
]
