
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="itazap/blt-1b-hf")
@strict
class BltLocalEncoderConfig(PreTrainedConfig):

    model_type = "blt_local_encoder"
    default_theta = 500000.0

    vocab_size: int = 260
    cross_attn_all_layers: bool | None = False
    cross_attn_k: int | None = 2
    hidden_size_global: int | None = 2048
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    num_hidden_layers: int = 1
    rms_norm_eps: float = 1e-5
    dropout: float | int | None = 0.0
    max_position_embeddings: int = 24576
    rope_parameters: RopeParameters | dict | None = None
    hidden_act: str = "silu"
    intermediate_size: int | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_key_value_heads or self.num_attention_heads
        self.intermediate_size = self.intermediate_size or int(8 * self.hidden_size / 3)
        self.tie_word_embeddings = False
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="itazap/blt-1b-hf")
@strict
class BltLocalDecoderConfig(PreTrainedConfig):

    model_type = "blt_local_decoder"
    default_theta = 500000.0

    vocab_size: int = 260
    cross_attn_all_layers: bool | None = True
    cross_attn_k: int | None = 2
    hidden_size_global: int | None = 2048
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    num_hidden_layers: int = 9
    rms_norm_eps: float = 1e-5
    dropout: float | int | None = 0.0
    max_position_embeddings: int = 24576
    rope_parameters: RopeParameters | dict | None = None
    hidden_act: str = "silu"
    intermediate_size: int = 2816
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_key_value_heads or self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = self.intermediate_size or int(8 * self.hidden_size / 3)
        self.tie_word_embeddings = False  # Force-set to False for BC
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="itazap/blt-1b-hf")
@strict
class BltGlobalTransformerConfig(PreTrainedConfig):
    model_type = "blt_global_transformer"
    default_theta = 500000.0

    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    num_hidden_layers: int = 25
    rms_norm_eps: float = 1e-5
    dropout: float | int | None = 0.0
    max_position_embeddings: int = 4096
    rope_parameters: RopeParameters | dict | None = None
    hidden_act: str = "silu"
    intermediate_size: int = 5632
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_key_value_heads or self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = self.intermediate_size or int(8 * self.hidden_size / 3)
        self.tie_word_embeddings = False

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="itazap/blt-1b-hf")
@strict
class BltPatcherConfig(PreTrainedConfig):
    model_type = "blt_patcher"

    vocab_size: int = 260
    hidden_size: int = 768
    num_hidden_layers: int = 14
    num_attention_heads: int = 12
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    dropout: float | int | None = 0.0
    intermediate_size: int = 2048
    rope_parameters: RopeParameters | dict | None = None
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_key_value_heads or self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = self.intermediate_size or int(8 * self.hidden_size / 3)
        self.tie_word_embeddings = False
        self.hidden_act = "silu"  # Blt uses silu activation

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="itazap/blt-1b-hf")
@strict
class BltConfig(PreTrainedConfig):

    model_type = "blt"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0
    sub_configs = {
        "patcher_config": BltPatcherConfig,
        "encoder_config": BltLocalEncoderConfig,
        "decoder_config": BltLocalDecoderConfig,
        "global_config": BltGlobalTransformerConfig,
    }

    vocab_size: int = 260
    max_position_embeddings: int = 4096
    patch_in_forward: bool | None = True
    patch_size: int | None = 4
    patching_mode: str | None = "entropy"
    patching_threshold: float | None = 1.335442066192627
    patching_batch_size: int | None = 1
    max_patch_length: int | None = None
    cross_attn_k: int | None = 2
    encoder_hash_byte_group_size: list[int] | None = None
    encoder_hash_byte_group_vocab: int | None = 500002
    encoder_hash_byte_group_nb_functions: int | None = 1
    patcher_config: dict | PreTrainedConfig | None = None
    encoder_config: dict | PreTrainedConfig | None = None
    decoder_config: dict | PreTrainedConfig | None = None
    global_config: dict | PreTrainedConfig | None = None
    tie_word_embeddings: bool = False
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    initializer_range: float = 0.02
    rope_parameters: RopeParameters | dict | None = None

    def __post_init__(self, **kwargs):
        self.encoder_hash_byte_group_size = self.encoder_hash_byte_group_size or [3, 4, 5, 6, 7, 8]

        if self.patcher_config is None:
            self.patcher_config = BltPatcherConfig(initializer_range=self.initializer_range)
            logger.info("patcher_config is None, using default Blt patcher config")
        elif isinstance(self.patcher_config, dict):
            self.patcher_config.setdefault("initializer_range", self.initializer_range)
            self.patcher_config = BltPatcherConfig(**self.patcher_config)

        if self.encoder_config is None:
            self.encoder_config = BltLocalEncoderConfig(initializer_range=self.initializer_range)
            logger.info("encoder_config is None, using default Blt encoder config")
        elif isinstance(self.encoder_config, dict):
            self.encoder_config.setdefault("initializer_range", self.initializer_range)
            self.encoder_config = BltLocalEncoderConfig(**self.encoder_config)

        if self.decoder_config is None:
            self.decoder_config = BltLocalDecoderConfig(initializer_range=self.initializer_range)
            logger.info("decoder_config is None, using default Blt decoder config")
        elif isinstance(self.decoder_config, dict):
            self.decoder_config.setdefault("initializer_range", self.initializer_range)
            self.decoder_config = BltLocalDecoderConfig(**self.decoder_config)

        if self.global_config is None:
            self.global_config = BltGlobalTransformerConfig(initializer_range=self.initializer_range)
            logger.info("global_config is None, using default Blt global config")
        elif isinstance(self.global_config, dict):
            self.global_config.setdefault("initializer_range", self.initializer_range)
            self.global_config = BltGlobalTransformerConfig(**self.global_config)

        encoder_cross_output_size = self.encoder_config.hidden_size * self.cross_attn_k
        self.global_config.encoder_cross_output_size = (
            encoder_cross_output_size if encoder_cross_output_size != self.global_config.hidden_size else None
        )

        super().__post_init__(**kwargs)


__all__ = [
    "BltConfig",
    "BltPatcherConfig",
    "BltLocalEncoderConfig",
    "BltLocalDecoderConfig",
    "BltGlobalTransformerConfig",
]
