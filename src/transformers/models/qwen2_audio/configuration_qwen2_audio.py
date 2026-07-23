
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="Qwen/Qwen2-Audio-7B")
@strict
class Qwen2AudioEncoderConfig(PreTrainedConfig):

    model_type = "qwen2_audio_encoder"
    attribute_map = {
        "num_hidden_layers": "encoder_layers",
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "intermediate_size": "encoder_ffn_dim",
    }

    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    encoder_layerdrop: float | int = 0.0
    d_model: int = 1280
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    activation_function: str = "gelu"
    activation_dropout: float | int = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500


@auto_docstring(checkpoint="Qwen/Qwen2-Audio-7B")
@strict
class Qwen2AudioConfig(PreTrainedConfig):

    model_type = "qwen2_audio"
    attribute_map = {
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_index: int = 151646

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "qwen2_audio_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["qwen2_audio_encoder"](
                d_model=1280,
                encoder_attention_heads=20,
                encoder_ffn_dim=5120,
                encoder_layerdrop=0.0,
                encoder_layers=32,
                num_mel_bins=128,
                max_source_positions=1500,
                scale_embedding=False,
                activation_function="gelu",
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        super().__post_init__(**kwargs)


__all__ = ["Qwen2AudioConfig", "Qwen2AudioEncoderConfig"]
