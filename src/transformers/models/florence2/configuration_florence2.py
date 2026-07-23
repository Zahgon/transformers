from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="florence-community/Florence-2-base")
@strict
class Florence2VisionConfig(PreTrainedConfig):

    model_type = "florence_vision"

    in_channels: int = 3
    depths: list[int] | tuple[int, ...] = (1, 1, 9, 1)
    patch_size: list[int] | tuple[int, ...] = (7, 3, 3, 3)
    patch_stride: list[int] | tuple[int, ...] = (4, 2, 2, 2)
    patch_padding: list[int] | tuple[int, ...] = (3, 1, 1, 1)
    patch_prenorm: list[bool] | tuple[bool, ...] = (False, True, True, True)
    embed_dim: list[int] | tuple[int, ...] = (128, 256, 512, 1024)
    num_heads: list[int] | tuple[int, ...] = (4, 8, 16, 32)
    num_groups: list[int] | tuple[int, ...] = (4, 8, 16, 32)
    window_size: int = 12
    drop_path_rate: float | int = 0.1
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    activation_function: str = "gelu"
    projection_dim: int = 1024
    max_temporal_embeddings: int = 100
    max_position_embeddings: int = 50
    initializer_range: float = 0.02


@auto_docstring(checkpoint="florence-community/Florence-2-base")
@strict
class Florence2Config(PreTrainedConfig):

    model_type = "florence2"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": Florence2VisionConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 51289
    is_encoder_decoder: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "bart")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["bart"]()

        if isinstance(self.vision_config, dict):
            self.vision_config = Florence2VisionConfig(**self.vision_config)
        elif self.vision_config is None:
            logger.info("vision_config is None. Initializing the Florence2VisionConfig with default values.")
            self.vision_config = Florence2VisionConfig()

        super().__post_init__(**kwargs)


__all__ = ["Florence2Config", "Florence2VisionConfig"]
