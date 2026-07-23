from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="nvidia/nemotron-3.5-asr-streaming-0.6b")
@strict
class Nemotron3_5AsrConfig(PreTrainedConfig):

    model_type = "nemotron3_5_asr"
    sub_configs = {"encoder_config": AutoConfig}

    vocab_size: int = 13088
    decoder_hidden_size: int = 640
    num_decoder_layers: int = 2
    hidden_act: str = "relu"
    max_symbols_per_step: int = 10
    encoder_config: dict | PreTrainedConfig | None = None
    pad_token_id: int = 0
    blank_token_id: int = 13087
    is_encoder_decoder: bool = True
    num_prompts: int = 128
    prompt_intermediate_size: int = 2048
    default_prompt_id: int = 101

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config["model_type"] = self.encoder_config.get("model_type", "nemotron_asr_streaming_encoder")
            self.encoder_config = CONFIG_MAPPING[self.encoder_config["model_type"]](**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = CONFIG_MAPPING["nemotron_asr_streaming_encoder"]()

        super().__post_init__(**kwargs)


__all__ = ["Nemotron3_5AsrConfig"]
