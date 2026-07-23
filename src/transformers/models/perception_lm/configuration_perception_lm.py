
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig
from ..timm_wrapper.configuration_timm_wrapper import TimmWrapperConfig


@auto_docstring(checkpoint="facebook/Perception-LM-1B")
@strict
class PerceptionLMConfig(PreTrainedConfig):

    model_type = "perception_lm"
    sub_configs = {"text_config": AutoConfig, "vision_config": TimmWrapperConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    vision_use_cls_token: bool = True
    projector_pooling_ratio: int = 1
    image_token_id: int = 128002
    video_token_id: int = 128003
    tie_word_embeddings: bool | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = TimmWrapperConfig(**self.vision_config)
        elif isinstance(self.vision_config, TimmWrapperConfig):
            pass
        elif self.vision_config is None:
            self.vision_config = TimmWrapperConfig()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["llama"]()

        if self.tie_word_embeddings is None:
            self.tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)

        super().__post_init__(**kwargs)


__all__ = ["PerceptionLMConfig"]
