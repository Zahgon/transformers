

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="google/shieldgemma-2-4b-it")
@strict
class ShieldGemma2Config(PreTrainedConfig):

    model_type = "shieldgemma2"
    attribute_map = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    mm_tokens_per_image: int = 256
    boi_token_index: int = 255_999
    eoi_token_index: int = 256_000
    image_token_index: int = 262_144
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "gemma3_text")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["gemma3_text"]()
        if kwargs.get("tie_word_embeddings") is None:
            self.tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", True)

        super().__post_init__(**kwargs)


__all__ = ["ShieldGemma2Config"]
