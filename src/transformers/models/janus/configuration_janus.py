
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="deepseek-community/Janus-Pro-1B")
@strict
class JanusVisionConfig(PreTrainedConfig):

    model_type = "janus_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 384
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    mlp_ratio: float | int = 4.0
    attention_bias: bool = True
    hidden_dropout_rate: float | int = 0.0
    projection_dim: int = 2048
    projection_dropout: float | int = 0.0
    use_qk_norm: bool = False
    initializer_range: float = 0.02
    depth: int = 2
    num_image_tokens: int = 576


@auto_docstring(checkpoint="deepseek-community/Janus-Pro-1B")
@strict
class JanusVQVAEConfig(PreTrainedConfig):

    model_type = "janus_vqgan"
    base_config_key = "vq_config"

    embed_dim: int = 8
    num_embeddings: int = 16384
    double_latent: bool = False
    latent_channels: int = 256
    in_channels: int = 3
    base_channels: int = 128
    channel_multiplier: list[int] | tuple[int, ...] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 2
    dropout: float | int = 0.0
    initializer_range: float = 0.02
    num_patches: int = 32
    out_channels: int = 3
    projection_dim: int = 2048
    num_hidden_layers: int = 2
    hidden_act: str = "gelu"
    image_token_embed_dim: int = 2048


@auto_docstring(checkpoint="deepseek-community/Janus-Pro-1B")
@strict
class JanusConfig(PreTrainedConfig):

    model_type = "janus"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": JanusVisionConfig,
        "vq_config": JanusVQVAEConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    vq_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 100581
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            logger.info("`text_config` is None. Initializing with default values")
            self.text_config = CONFIG_MAPPING["llama"]()

        if self.vision_config is None:
            logger.info("`vision_config` is None. Initializing with default JanusVisionConfig values")
            self.vision_config = JanusVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = JanusVisionConfig(**self.vision_config)

        if self.vq_config is None:
            logger.info("`vq_config` is None. Initializing with default JanusVQVAEConfig values")
            self.vq_config = JanusVQVAEConfig()
        elif isinstance(self.vq_config, dict):
            self.vq_config = JanusVQVAEConfig(**self.vq_config)

        self.vq_config.num_patches = self.vision_config.image_size // self.vision_config.patch_size
        super().__post_init__(**kwargs)


__all__ = ["JanusVQVAEConfig", "JanusVisionConfig", "JanusConfig"]
