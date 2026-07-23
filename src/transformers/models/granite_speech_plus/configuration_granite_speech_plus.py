from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-plus")
@strict
class GraniteSpeechPlusEncoderConfig(PreTrainedConfig):

    model_type = "granite_speech_plus_encoder"
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

    cat_hidden_layers: list[int] | None = None

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.dim_head is None:
            self.dim_head = self.hidden_dim // self.num_heads


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-plus")
@strict
class GraniteSpeechPlusConfig(PreTrainedConfig):

    model_type = "granite_speech_plus"
    attribute_map = {
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {
        "text_config": AutoConfig,
        "encoder_config": GraniteSpeechPlusEncoderConfig,
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

        if not isinstance(self.encoder_config, GraniteSpeechPlusEncoderConfig):
            self.encoder_config = {} if self.encoder_config is None else self.encoder_config
            self.encoder_config = GraniteSpeechPlusEncoderConfig(**self.encoder_config)

        super().__post_init__(**kwargs)

        if self.encoder_config.cat_hidden_layers is not None:
            for idx in self.encoder_config.cat_hidden_layers:
                if idx < 0 or idx >= self.encoder_config.num_layers:
                    raise ValueError(
                        f"cat_hidden_layers index {idx} is out of range [0, {self.encoder_config.num_layers})."
                    )
        if self.encoder_config.cat_hidden_layers is not None:
            num_concat = len(self.encoder_config.cat_hidden_layers) + 1
            if self.projector_config.encoder_hidden_size != self.encoder_config.hidden_dim * num_concat:
                raise ValueError(
                    f"projector encoder_hidden_size {self.projector_config.encoder_hidden_size} "
                    f"must equal encoder hidden_dim * {num_concat} = "
                    f"{self.encoder_config.hidden_dim * num_concat}."
                )


__all__ = ["GraniteSpeechPlusConfig", "GraniteSpeechPlusEncoderConfig"]
