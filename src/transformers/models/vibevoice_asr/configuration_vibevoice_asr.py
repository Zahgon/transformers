import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="microsoft/VibeVoice-ASR-HF")
@strict
class VibeVoiceAsrConfig(PreTrainedConfig):

    model_type = "vibevoice_asr"
    sub_configs = {
        "acoustic_tokenizer_encoder_config": AutoConfig,
        "semantic_tokenizer_encoder_config": AutoConfig,
        "text_config": AutoConfig,
    }

    acoustic_tokenizer_encoder_config: dict | PreTrainedConfig | None = None
    semantic_tokenizer_encoder_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151648
    audio_bos_token_id: int = 151646
    audio_eos_token_id: int = 151647
    acoustic_tokenizer_chunk_size: int = 1440000

    def __post_init__(self, **kwargs):
        if isinstance(self.acoustic_tokenizer_encoder_config, dict):
            self.acoustic_tokenizer_encoder_config["model_type"] = self.acoustic_tokenizer_encoder_config.get(
                "model_type", "vibevoice_acoustic_tokenizer_encoder"
            )
            self.acoustic_tokenizer_encoder_config = CONFIG_MAPPING[
                self.acoustic_tokenizer_encoder_config["model_type"]
            ](**self.acoustic_tokenizer_encoder_config)
        elif self.acoustic_tokenizer_encoder_config is None:
            self.acoustic_tokenizer_encoder_config = CONFIG_MAPPING["vibevoice_acoustic_tokenizer_encoder"]()

        if isinstance(self.semantic_tokenizer_encoder_config, dict):
            self.semantic_tokenizer_encoder_config["model_type"] = self.semantic_tokenizer_encoder_config.get(
                "model_type", "vibevoice_acoustic_tokenizer_encoder"
            )
            self.semantic_tokenizer_encoder_config = CONFIG_MAPPING[
                self.semantic_tokenizer_encoder_config["model_type"]
            ](**self.semantic_tokenizer_encoder_config)
        elif self.semantic_tokenizer_encoder_config is None:
            self.semantic_tokenizer_encoder_config = CONFIG_MAPPING["vibevoice_acoustic_tokenizer_encoder"](
                hidden_size=128
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def max_position_embeddings(self) -> int:
        pass

    @max_position_embeddings.setter
    def max_position_embeddings(self, value: int):
        pass


__all__ = ["VibeVoiceAsrConfig"]
