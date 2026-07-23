

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
@strict
class SmolVLMVisionConfig(PreTrainedConfig):

    model_type = "smolvlm_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1152
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 32
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
@strict
class SmolVLMConfig(PreTrainedConfig):

    model_type = "smolvlm"
    sub_configs = {"text_config": AutoConfig, "vision_config": SmolVLMVisionConfig}

    use_cache: bool = True
    image_token_id: int = 128257
    tie_word_embeddings: bool = False
    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    scale_factor: int = 2
    pad_token_id: int | None = 128_002

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = SmolVLMVisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(self.vision_config, dict):
            self.vision_config = SmolVLMVisionConfig(**self.vision_config)

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            logger.info("text_config is None, using default Llama text config")
            self.text_config = CONFIG_MAPPING["llama"](
                rms_norm_eps=1e-5,
                pad_token_id=self.pad_token_id,
            )

        super().__post_init__(**kwargs)


__all__ = ["SmolVLMVisionConfig", "SmolVLMConfig"]
