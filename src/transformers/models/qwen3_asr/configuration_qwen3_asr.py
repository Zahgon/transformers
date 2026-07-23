
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="Qwen/Qwen3-ASR-1.7B-hf")
@strict
class Qwen3ASREncoderConfig(PreTrainedConfig):

    model_type = "qwen3_asr_encoder"
    attribute_map = {
        "num_hidden_layers": "encoder_layers",
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "intermediate_size": "encoder_ffn_dim",
    }

    num_mel_bins: int = 128
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    d_model: int = 1024
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    activation_function: str = "gelu"
    activation_dropout: float | int = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02

    n_window: int = 50
    output_dim: int = 3584
    n_window_infer: int = 800
    downsample_hidden_size: int = 480
    max_position_embeddings: int = 13


@auto_docstring(checkpoint="Qwen/Qwen3-ASR-1.7B-hf")
@strict
class Qwen3ASRConfig(PreTrainedConfig):

    model_type = "qwen3_asr"
    sub_configs = {"audio_config": AutoConfig, "text_config": AutoConfig}

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151676
    timestamp_token_id: int = 151705
    pad_token_id: int = 151645
    eos_token_id: list[int] | tuple[int, ...] | int = (151643, 151645)
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    token_classification_bias: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "qwen3_asr_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["qwen3_asr_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"](
                hidden_size=2048,
                intermediate_size=6144,
                num_hidden_layers=28,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=128,
                max_position_embeddings=65536,
                tie_word_embeddings=True,
            )

        super().__post_init__(**kwargs)


__all__ = ["Qwen3ASREncoderConfig", "Qwen3ASRConfig"]
