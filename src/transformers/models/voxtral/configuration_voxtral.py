

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="mistralai/Voxtral-Mini-3B-2507")
@strict
class VoxtralEncoderConfig(PreTrainedConfig):

    model_type = "voxtral_encoder"

    attribute_map = {
        "d_model": "hidden_size",
        "encoder_layers": "num_hidden_layers",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
        "encoder_layerdrop": "layerdrop",
    }

    vocab_size: int = 51866
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    scale_embedding: bool = False
    activation_function: str = "gelu"
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0

    dropout: float | int = 0.0
    layerdrop: float | int = 0.0
    activation_dropout: float | int = 0.0


@auto_docstring(checkpoint="mistralai/Voxtral-Mini-3B-2507")
@strict
class VoxtralConfig(PreTrainedConfig):

    model_type = "voxtral"
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    _default_text_config_kwargs = {
        "vocab_size": 131072,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 30,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-05,
        "use_cache": True,
        "rope_theta": 100000000.0,
        "head_dim": 128,
    }

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int | None = None
    projector_hidden_act: str = "gelu"

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "voxtral_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["voxtral_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["llama"](**self._default_text_config_kwargs)

        self.hidden_size = self.text_config.hidden_size
        super().__post_init__(**kwargs)


__all__ = ["VoxtralEncoderConfig", "VoxtralConfig"]
