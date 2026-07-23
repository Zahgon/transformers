
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="deepseek-community/deepseek-vl-7b-chat")
@strict
class DeepseekVLHybridConfig(PreTrainedConfig):

    model_type = "deepseek_vl_hybrid"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig, "high_res_vision_config": AutoConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 100015
    tie_word_embeddings: bool = True

    high_res_vision_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.high_res_vision_config is None:
            self.high_res_vision_config = {}
            logger.info("`high_res_vision_config` is `None`. Initializing the `SamVisionConfig` with default values.")

        if isinstance(self.high_res_vision_config, dict):
            self.high_res_vision_config["model_type"] = self.high_res_vision_config.get(
                "model_type", "sam_vision_model"
            )
            self.high_res_vision_config = CONFIG_MAPPING[self.high_res_vision_config["model_type"]](
                **self.high_res_vision_config
            )
        if self.text_config is None:
            self.text_config = {}
            logger.info("`text_config` is `None`. Initializing the `LlamaConfig` with default values.")
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)

        if self.vision_config is None:
            self.vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `SiglipVisionConfig` with default values.")
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["DeepseekVLHybridConfig"]
