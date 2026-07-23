

import numpy as np
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ...video_utils import VideoMetadata
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..glm4v.image_processing_glm4v import Glm4vImageProcessor
from ..glm4v.image_processing_pil_glm4v import Glm4vImageProcessorPil
from ..glm4v.modeling_glm4v import Glm4vForConditionalGeneration, Glm4vModel, Glm4vPreTrainedModel
from ..glm4v.processing_glm4v import Glm4vProcessor
from ..glm4v.video_processing_glm4v import Glm4vVideoProcessor


@auto_docstring(checkpoint="zai-org/GLM-4.1V-9B-Thinking")
@strict
class Glm46VConfig(PreTrainedConfig):

    model_type = "glm46v"
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


class Glm46VPreTrainedModel(Glm4vPreTrainedModel):
    _can_record_outputs = None
    _no_split_modules = None

    def _init_weights(self, module):
        raise AttributeError("Not needed")


class Glm46VModel(Glm4vModel):
    _no_split_modules = None

    def __init__(self, config):
        super().__init__(config)
        self.visual = AutoModel.from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)


class Glm46VForConditionalGeneration(Glm4vForConditionalGeneration):
    pass


class Glm46VProcessor(Glm4vProcessor):
    def replace_frame_token_id(self, timestamp_sec, num_image_tokens: int = 1):
        pass


class Glm46VImageProcessorPil(Glm4vImageProcessorPil):
    pass


class Glm46VImageProcessor(Glm4vImageProcessor):
    pass


class Glm46VVideoProcessor(Glm4vVideoProcessor):
    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: int | float | None = None,
        **kwargs,
    ):
        pass


__all__ = [
    "Glm46VConfig",
    "Glm46VModel",
    "Glm46VPreTrainedModel",
    "Glm46VForConditionalGeneration",
    "Glm46VProcessor",
    "Glm46VImageProcessor",
    "Glm46VImageProcessorPil",
    "Glm46VVideoProcessor",
]
