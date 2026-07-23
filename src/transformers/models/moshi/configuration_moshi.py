
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="kmhf/hf-moshiko")
@strict
class MoshiDepthConfig(PreTrainedConfig):

    model_type = "moshi_depth"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32000
    hidden_size: int = 1024
    input_size: int = 4096
    num_hidden_layers: int = 6
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    audio_vocab_size: int = 2048
    max_position_embeddings: int = 9
    hidden_act: str = "silu"
    head_dim: int | None = None
    initializer_range: float = 0.02
    use_cache: bool = True
    sliding_window: int = 8
    attention_dropout: float | int = 0.0
    ffn_dim: int = 5632
    rms_norm_eps: float = 1e-8
    num_codebooks: int = 8
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = (
            self.num_key_value_heads if self.num_key_value_heads is not None else self.num_attention_heads
        )
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="kmhf/hf-moshiko")
@strict
class MoshiConfig(PreTrainedConfig):

    model_type = "moshi"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"audio_encoder_config": AutoConfig, "depth_decoder_config": MoshiDepthConfig}

    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    audio_vocab_size: int | None = None
    max_position_embeddings: int = 3000
    rope_parameters: RopeParameters | dict | None = None
    hidden_act: str = "silu"
    head_dim: int | None = None
    initializer_range: float = 0.02
    use_cache: bool = True
    sliding_window: int = 3000
    attention_dropout: float | int = 0.0
    ffn_dim: int = 22528
    rms_norm_eps: float = 1e-8
    num_codebooks: int = 8
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    audio_encoder_config: dict | PreTrainedConfig | None = None
    depth_decoder_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = (
            self.num_key_value_heads if self.num_key_value_heads is not None else self.num_attention_heads
        )
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads

        if isinstance(self.audio_encoder_config, dict):
            audio_encoder_model_type = self.audio_encoder_config.pop("model_type", "mimi")
            self.audio_encoder_config = AutoConfig.for_model(audio_encoder_model_type, **self.audio_encoder_config)
        elif self.audio_encoder_config is None:
            self.audio_encoder_config = AutoConfig.for_model("mimi")

        self.audio_vocab_size = (
            self.audio_encoder_config.codebook_size if self.audio_vocab_size is None else self.audio_vocab_size
        )

        if isinstance(self.depth_decoder_config, dict):
            self.depth_decoder_config.update(
                {
                    "audio_vocab_size": self.audio_vocab_size,
                    "input_size": self.hidden_size,
                    "vocab_size": self.vocab_size,
                    "num_codebooks": self.num_codebooks,
                }
            )
            self.depth_decoder_config = MoshiDepthConfig(**self.depth_decoder_config)
        elif self.depth_decoder_config is None:
            self.depth_decoder_config = MoshiDepthConfig()
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def sampling_rate(self):
        pass

    @classmethod
    def from_audio_encoder_config(
        cls,
        audio_encoder_config: PreTrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MoshiConfig`] (or a derived class) from an audio encoder configuration.

        Returns:
            [`MoshiConfig`]: An instance of a configuration object
        """

        return cls(
            audio_encoder_config=audio_encoder_config.to_dict(),
            **kwargs,
        )


__all__ = ["MoshiConfig", "MoshiDepthConfig"]
