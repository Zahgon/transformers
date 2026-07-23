
import numpy as np
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ...utils.type_validators import interval
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="HKUSTAudio/xcodec2-hf")
@strict
class Xcodec2Config(PreTrainedConfig):

    model_type = "xcodec2"
    keys_to_ignore_at_inference = ["past_key_values"]
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)
    rms_norm_eps: float = 1e-6
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    head_dim: int = 64
    sub_configs = {"semantic_model_config": AutoConfig}

    encoder_hidden_size: int = 48
    downsampling_ratios: list[int] | tuple[int, ...] = (2, 2, 4, 4, 5)
    semantic_model_config: dict | PreTrainedConfig | None = None
    sampling_rate: int = 16000
    activation_dropout: float = 0.1
    quantization_dim: int = 2048
    quantization_levels: list[int] | tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4, 4)

    def __post_init__(self, **kwargs):
        if isinstance(self.semantic_model_config, dict):
            self.semantic_model_config["model_type"] = self.semantic_model_config.get("model_type", "wav2vec2-bert")
            self.semantic_model_config = CONFIG_MAPPING[self.semantic_model_config["model_type"]](
                **self.semantic_model_config
            )
        elif self.semantic_model_config is None:
            self.semantic_model_config = CONFIG_MAPPING["wav2vec2-bert"](num_hidden_layers=16)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def hop_length(self) -> int:
        pass

    @property
    def n_fft(self) -> int:
        pass


__all__ = ["Xcodec2Config"]
