
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="ModernVBERT/modernvbert")
@strict
class ModernVBertConfig(PreTrainedConfig):

    model_type = "modernvbert"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    text_config: PreTrainedConfig | dict | None = None
    vision_config: PreTrainedConfig | dict | None = None
    image_token_id: int = 50407
    pixel_shuffle_factor: int = 4
    initializer_range: float = 0.02
    initializer_cutoff_factor: float = 2.0
    classifier_pooling: Literal["cls", "mean"] = "cls"
    classifier_dropout: float | int = 0.0
    classifier_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = CONFIG_MAPPING["modernbert"]()
        elif isinstance(self.text_config, dict):
            self.text_config = CONFIG_MAPPING["modernbert"](**self.text_config)

        if self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"]()
        elif isinstance(self.vision_config, dict):
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["ModernVBertConfig"]
