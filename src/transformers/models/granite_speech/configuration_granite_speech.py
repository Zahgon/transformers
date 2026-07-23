
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="ibm-granite/granite-speech-3.3-2b")
@strict
class GraniteSpeechEncoderConfig(PreTrainedConfig):

    model_type = "granite_speech_encoder"
    attribute_map = {
        "hidden_size": "hidden_dim",
        "num_hidden_layers": "num_layers",
        "num_attention_heads": "num_heads",
        "num_mel_bins": "input_dim",
    }

    input_dim: int = 160
    num_layers: int = 10
    hidden_dim: int = 1024
    feedforward_mult: int = 4
    num_heads: int = 8
    dim_head: int | None = None
    output_dim: int = 42
    context_size: int = 200
    max_pos_emb: int = 512
    dropout: float | int = 0.1
    conv_kernel_size: int = 15
    conv_expansion_factor: int = 2

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.dim_head is None:
            self.dim_head = self.hidden_dim // self.num_heads


@auto_docstring(checkpoint="ibm-granite/granite-speech-3.3-2b")
@strict
class GraniteSpeechConfig(PreTrainedConfig):

    model_type = "granite_speech"
    attribute_map = {
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {
        "text_config": AutoConfig,
        "encoder_config": GraniteSpeechEncoderConfig,
        "projector_config": AutoConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    encoder_config: dict | PreTrainedConfig | None = None
    projector_config: dict | PreTrainedConfig | None = None
    audio_token_index: int = 49155
    initializer_range: float = 0.02
    has_lora_adapter: bool = True
    downsample_rate: int = 5
    window_size: int = 15
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "granite")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["granite"]()

        if isinstance(self.projector_config, dict):
            self.projector_config["model_type"] = self.projector_config.get("model_type", "blip_2_qformer")
            self.projector_config = CONFIG_MAPPING[self.projector_config["model_type"]](**self.projector_config)
        elif self.projector_config is None:
            self.projector_config = CONFIG_MAPPING["blip_2_qformer"]()

        if not isinstance(self.encoder_config, GraniteSpeechEncoderConfig):
            self.encoder_config = {} if self.encoder_config is None else self.encoder_config
            self.encoder_config = GraniteSpeechEncoderConfig(**self.encoder_config)

        super().__post_init__(**kwargs)


__all__ = ["GraniteSpeechEncoderConfig", "GraniteSpeechConfig"]
