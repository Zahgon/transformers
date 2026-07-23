

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="facebook/pe-av-large")
@strict
class PeAudioVideoEncoderConfig(PreTrainedConfig):

    model_type = "pe_audio_video_encoder"
    base_config_key = "audio_video_config"
    sub_configs = {"audio_config": AutoConfig, "video_config": AutoConfig}

    audio_config: dict | PreTrainedConfig | None = None
    video_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 1792
    intermediate_size: int = 4800
    num_hidden_layers: int = 6
    num_attention_heads: int = 14
    num_key_value_heads: int | None = None
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 10000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "pe_audio_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["pe_audio_encoder"]()

        if isinstance(self.video_config, dict):
            self.video_config["model_type"] = self.video_config.get("model_type", "pe_video_encoder")
            self.video_config = CONFIG_MAPPING[self.video_config["model_type"]](**self.video_config)
        elif self.video_config is None:
            self.video_config = CONFIG_MAPPING["pe_video_encoder"]()

        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 20000}
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/pe-av-large")
@strict
class PeAudioVideoConfig(PreTrainedConfig):

    model_type = "pe_audio_video"
    sub_configs = {"text_config": AutoConfig, "audio_video_config": PeAudioVideoEncoderConfig}

    _default_text_config_kwargs = {
        "model_type": "modernbert",
        "hidden_size": 1024,
        "intermediate_size": 2624,
        "num_hidden_layers": 22,
        "num_attention_heads": 16,
    }

    text_config: dict | PreTrainedConfig | None = None
    audio_video_config: dict | PreTrainedConfig | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "modernbert")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["modernbert"](**self._default_text_config_kwargs)

        if isinstance(self.audio_video_config, dict):
            self.audio_video_config = PeAudioVideoEncoderConfig(**self.audio_video_config)
        elif self.audio_video_config is None:
            self.audio_video_config = PeAudioVideoEncoderConfig()

        super().__post_init__(**kwargs)

    @property
    def audio_config(self):
        pass

    @property
    def video_config(self):
        pass


__all__ = ["PeAudioVideoEncoderConfig", "PeAudioVideoConfig"]
