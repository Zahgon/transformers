

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING


@auto_docstring(checkpoint="UsefulSensors/moonshine-streaming-tiny")
@strict
class MoonshineStreamingEncoderConfig(PreTrainedConfig):

    model_type = "moonshine_streaming_encoder"

    hidden_size: int = 320
    intermediate_size: int = 1280
    hidden_act: str = "gelu"
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    max_position_embeddings: int = 4096
    attention_dropout: float | int = 0.0
    attention_bias: bool = False
    sample_rate: int = 16000
    frame_ms: float = 5.0
    sliding_windows: tuple[tuple[int, int], ...] | list[list[int, int]] = (
        (16, 4),
        (16, 4),
        (16, 0),
        (16, 0),
        (16, 4),
        (16, 4),
    )
    head_dim: int | None = None

    def __post_init__(self, **kwargs):
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        self.sliding_windows = [list(window) for window in self.sliding_windows]

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="UsefulSensors/moonshine-streaming-tiny")
@strict
class MoonshineStreamingConfig(PreTrainedConfig):

    model_type = "moonshine_streaming"
    sub_configs = {"encoder_config": MoonshineStreamingEncoderConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    encoder_config: dict | MoonshineStreamingEncoderConfig | None = None
    vocab_size: int = 32768
    hidden_size: int = 320
    intermediate_size: int = 1280
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    decoder_start_token_id: int | None = None
    head_dim: int | None = None
    pad_head_dim_to_multiple_of: int | None = None
    tie_word_embeddings: bool = False
    is_encoder_decoder: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config["model_type"] = self.encoder_config.get("model_type", "moonshine_streaming_encoder")
            self.encoder_config = CONFIG_MAPPING[self.encoder_config["model_type"]](**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = CONFIG_MAPPING["moonshine_streaming_encoder"]()

        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.8,
            }
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["MoonshineStreamingConfig", "MoonshineStreamingEncoderConfig"]
