
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="HuggingFaceM4/idefics2-8b")
@strict
class Idefics2VisionConfig(PreTrainedConfig):

    model_type = "idefics2_vision"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 32
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="HuggingFaceM4/idefics2-8b")
@strict
class Idefics2PerceiverConfig(PreTrainedConfig):

    model_type = "idefics2_perceiver"

    hidden_act: str = "silu"
    hidden_size: int = 4096
    rms_norm_eps: float = 1e-06
    resampler_n_latents: int = 64
    resampler_depth: int = 3
    resampler_n_heads: int = 16
    resampler_head_dim: int = 96
    num_key_value_heads: int = 4
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="HuggingFaceM4/idefics2-8b")
@strict
class Idefics2Config(PreTrainedConfig):

    model_type = "idefics2"
    sub_configs = {
        "text_config": AutoConfig,
        "perceiver_config": Idefics2PerceiverConfig,
        "vision_config": Idefics2VisionConfig,
    }

    use_cache: bool = True
    image_token_id: int = 32_001
    tie_word_embeddings: bool = False
    vision_config: dict | PreTrainedConfig | None = None
    perceiver_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.perceiver_config is None:
            self.perceiver_config = Idefics2PerceiverConfig()
            logger.info("perciver_config is None, using default perceiver config")
        elif isinstance(self.perceiver_config, dict):
            self.perceiver_config = Idefics2PerceiverConfig(**self.perceiver_config)

        if self.vision_config is None:
            self.vision_config = Idefics2VisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Idefics2VisionConfig(**self.vision_config)

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "mistral")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            logger.info("text_config is None, using default text config")
            self.text_config = CONFIG_MAPPING["mistral"](
                max_position_embeddings=4096 * 8,
                rms_norm_eps=1e-5,
                pad_token_id=0,
            )

        if self.text_config.hidden_size != self.perceiver_config.hidden_size:
            self.perceiver_config.hidden_size = self.text_config.hidden_size
            self.perceiver_config.rms_norm_eps = self.text_config.rms_norm_eps
            logger.warning_once(
                "Perceiver config has a different `hidden_size` than text config, which means default values were used. "
                "In your model's config on the hub, add `hidden_size` and `rms_norm_eps` keys under the `perceiver_config` dict. "
            )

        super().__post_init__(**kwargs)


__all__ = ["Idefics2Config", "Idefics2PerceiverConfig", "Idefics2VisionConfig"]
