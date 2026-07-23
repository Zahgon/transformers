
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="llava-hf/llava-onevision-qwen2-7b-ov-hf")
@strict
class LlavaOnevisionConfig(PreTrainedConfig):

    model_type = "llava_onevision"
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 151646
    video_token_index: int = 151647
    projector_hidden_act: str = "gelu"
    vision_feature_select_strategy: Literal["default", "full"] = "full"
    vision_feature_layer: int | list[int] = -1
    multimodal_projector_bias: bool = True
    tie_word_embeddings: bool = False
    image_grid_pinpoints: list | None = None
    vision_aspect_ratio: str = "anyres_max_9"

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                hidden_size=1152,
                intermediate_size=4304,
                patch_size=14,
                image_size=384,
                num_hidden_layers=26,
                num_attention_heads=16,
                vision_use_head=False,
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        self.image_grid_pinpoints = (
            self.image_grid_pinpoints
            if self.image_grid_pinpoints is not None
            else [
                [384, 384],
                [384, 768],
                [384, 1152],
                [384, 1536],
                [384, 1920],
                [384, 2304],
                [768, 384],
                [768, 768],
                [768, 1152],
                [768, 1536],
                [768, 1920],
                [768, 2304],
                [1152, 384],
                [1152, 768],
                [1152, 1152],
                [1152, 1536],
                [1152, 1920],
                [1152, 2304],
                [1536, 384],
                [1536, 768],
                [1536, 1152],
                [1536, 1536],
                [1536, 1920],
                [1536, 2304],
                [1920, 384],
                [1920, 768],
                [1920, 1152],
                [1920, 1536],
                [1920, 1920],
                [1920, 2304],
                [2304, 384],
                [2304, 768],
                [2304, 1152],
                [2304, 1536],
                [2304, 1920],
                [2304, 2304],
            ]
        )

        if not self.tie_word_embeddings and self.text_config.tie_word_embeddings:
            self.tie_word_embeddings = self.text_config.tie_word_embeddings

        super().__post_init__(**kwargs)


__all__ = ["LlavaOnevisionConfig"]
