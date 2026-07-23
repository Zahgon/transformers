
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="zai-org/GLM-4.1V-9B-Thinking")
@strict
class GlmgaConfig(PreTrainedConfig):

    model_type = "glmga"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151343
    video_token_id: int = 151344
    image_start_token_id: int = 151339
    image_end_token_id: int = 151340
    video_start_token_id: int = 151361
    video_end_token_id: int = 151362
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "glm4v_vision")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["glm4v_vision"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "glm4v_text")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["glm4v_text"]()

        super().__post_init__(**kwargs)


__all__ = ["GlmgaConfig"]
