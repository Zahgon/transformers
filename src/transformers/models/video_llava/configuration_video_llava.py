
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="LanguageBind/Video-LLaVA-7B-hf")
@strict
class VideoLlavaConfig(PreTrainedConfig):

    model_type = "video_llava"
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 32000
    video_token_index: int = 32001
    projector_hidden_act: str = "gelu"
    vision_feature_select_strategy: Literal["default", "full"] = "default"
    vision_feature_layer: int | list[int] = -2
    image_seq_length: int = 256
    video_seq_length: int = 2056
    multimodal_projector_bias: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            if "model_type" not in self.vision_config:
                self.vision_config["model_type"] = "clip_vision_model"
                logger.warning("Key=`model_type` not found in vision config, setting it to `clip_vision_model`")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=224,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        if isinstance(self.text_config, dict):
            if "model_type" not in self.text_config:
                self.text_config["model_type"] = "llama"
                logger.warning("Key=`model_type` not found in text config, setting it to `llama`")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["llama"]()

        super().__post_init__(**kwargs)


__all__ = ["VideoLlavaConfig"]
