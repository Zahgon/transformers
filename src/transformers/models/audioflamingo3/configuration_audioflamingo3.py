

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="nvidia/audio-flamingo-3-hf")
@strict
class AudioFlamingo3EncoderConfig(PreTrainedConfig):

    model_type = "audioflamingo3_encoder"

    attribute_map = {
        "d_model": "hidden_size",
        "encoder_layers": "num_hidden_layers",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
        "encoder_layerdrop": "layerdrop",
    }

    num_mel_bins: int = 128
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    intermediate_size: int = 5120
    layerdrop: float | int = 0.0
    activation_function: str = "gelu"
    hidden_size: int = 1280
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    initializer_range: float = 0.02
    scale_embedding: bool = False
    max_source_positions: int = 1500


@auto_docstring(checkpoint="nvidia/audio-flamingo-3-hf")
@strict
class AudioFlamingo3Config(PreTrainedConfig):

    model_type = "audioflamingo3"
    sub_configs = {"audio_config": AutoConfig, "text_config": AutoConfig}
    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151669
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "audioflamingo3_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["audioflamingo3_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        super().__post_init__(**kwargs)


__all__ = ["AudioFlamingo3Config", "AudioFlamingo3EncoderConfig"]
