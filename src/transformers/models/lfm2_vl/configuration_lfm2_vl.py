
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="LiquidAI/LFM2-VL-1.6B")
@strict
class Lfm2VlConfig(PreTrainedConfig):

    model_type = "lfm2_vl"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 396
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 2560
    projector_bias: bool = True
    projector_use_layernorm: bool = True
    downsample_factor: int = 2
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "siglip2_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip2_vision_model"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "lfm2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["lfm2"]()

        self.tie_word_embeddings = kwargs.pop("tie_embedding", self.tie_word_embeddings)
        super().__post_init__(**kwargs)


__all__ = ["Lfm2VlConfig"]
