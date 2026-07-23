

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="nvidia/music-flamingo-2601-hf")
@strict
class MusicFlamingoConfig(PreTrainedConfig):

    model_type = "musicflamingo"
    sub_configs = {"audio_config": AutoConfig, "text_config": AutoConfig}
    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151669
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True

    audio_bos_token_id: int = 151670
    audio_eos_token_id: int = 151671
    audio_frame_step: float = 0.01
    rope_parameters: dict | None = None

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": 1200.0,
                "partial_rotary_factor": 0.2,
            }
        if isinstance(self.audio_config, dict):
            if self.audio_config["model_type"] in [None, "musicflamingo_encoder"]:
                self.audio_config["model_type"] = "audioflamingo3_encoder"

            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["audioflamingo3_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        self.max_position_embeddings = self.rope_parameters["rope_theta"]
        self.head_dim = self.audio_config.hidden_size
        super().__post_init__(**kwargs)


__all__ = ["MusicFlamingoConfig"]
