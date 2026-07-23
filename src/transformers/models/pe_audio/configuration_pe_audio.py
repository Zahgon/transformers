

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="facebook/pe-av-large")
@strict
class PeAudioEncoderConfig(PreTrainedConfig):

    model_type = "pe_audio_encoder"
    sub_configs = {"dac_config": AutoConfig}
    base_config_key = "audio_video_config"

    _default_dac_config_kwargs = {
        "downsampling_ratios": [2, 8, 10, 12],
        "encoder_hidden_size": 64,
        "codebook_dim": 128,
    }
    dac_config: dict | PreTrainedConfig | None = None
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

        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 20000, "rope_type": "default"}

        if isinstance(self.dac_config, dict):
            self.dac_config["model_type"] = self.dac_config.get("model_type", "dac")
            self.dac_config = CONFIG_MAPPING[self.dac_config["model_type"]](
                **{**self._default_dac_config_kwargs, **self.dac_config}
            )
        elif self.dac_config is None:
            self.dac_config = CONFIG_MAPPING["dac"](**self._default_dac_config_kwargs)

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/pe-av-large")
@strict
class PeAudioConfig(PreTrainedConfig):

    model_type = "pe_audio"
    sub_configs = {"text_config": AutoConfig, "audio_config": PeAudioEncoderConfig}
    base_config_key = "audio_video_config"

    _default_text_config_kwargs = {
        "model_type": "modernbert",
        "hidden_size": 1024,
        "intermediate_size": 2624,
        "num_hidden_layers": 22,
        "num_attention_heads": 16,
    }

    text_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "modernbert")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["modernbert"](**self._default_text_config_kwargs)

        if isinstance(self.audio_config, dict):
            self.audio_config = PeAudioEncoderConfig(**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = PeAudioEncoderConfig()
        super().__post_init__(**kwargs)


__all__ = ["PeAudioEncoderConfig", "PeAudioConfig"]
