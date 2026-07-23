
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="nvidia/Cosmos3-Nano")
@strict
class Cosmos3OmniConfig(PreTrainedConfig):

    model_type = "cosmos3_omni"
    sub_configs = {"vision_config": AutoConfig, "text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            model_type = self.vision_config.pop("model_type", "qwen3_vl_vision")
            if model_type == "qwen3_vl":
                model_type = "qwen3_vl_vision"
            self.vision_config = CONFIG_MAPPING[model_type](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["qwen3_vl_vision"]()

        if isinstance(self.text_config, dict):
            model_type = self.text_config.get("model_type", "qwen3_vl_text")
            self.text_config = CONFIG_MAPPING[model_type](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3_vl_text"]()

        super().__post_init__(**kwargs)


__all__ = ["Cosmos3OmniConfig"]
