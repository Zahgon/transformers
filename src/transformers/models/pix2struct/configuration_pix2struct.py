
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/pix2struct-base")
@strict
class Pix2StructTextConfig(PreTrainedConfig):

    model_type = "pix2struct_text_model"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "decoder_attention_heads": "num_heads",
        "encoder_attention_heads": "num_heads",
        "encoder_layers": "num_layers",
        "decoder_layers": "num_layers",
    }

    vocab_size: int = 50244
    hidden_size: int = 768
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 12
    num_heads: int = 12
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float | int = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    dense_act_fn: str = "gelu_new"
    decoder_start_token_id: int = 0
    use_cache: bool = False
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = None
    tie_word_embeddings: bool = False
    is_decoder: bool = True
    add_cross_attention: bool = False


@auto_docstring(checkpoint="google/pix2struct-base")
@strict
class Pix2StructVisionConfig(PreTrainedConfig):

    model_type = "pix2struct_vision_model"

    hidden_size: int = 768
    patch_embed_hidden_size: int = 768
    d_ff: int = 2048
    d_kv: int = 64
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    dense_act_fn: str = "gelu_new"
    layer_norm_eps: float = 1e-6
    dropout_rate: float | int = 0.0
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10
    initializer_factor: float = 1.0
    seq_len: int = 4096
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128


@auto_docstring(checkpoint="google/pix2struct-base")
@strict
class Pix2StructConfig(PreTrainedConfig):

    model_type = "pix2struct"
    sub_configs = {"text_config": Pix2StructTextConfig, "vision_config": Pix2StructVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    initializer_factor: float = 1.0
    initializer_range: float = 0.02
    is_vqa: bool = False
    tie_word_embeddings: bool = False
    is_encoder_decoder: bool = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Pix2StructTextConfig(
                is_encoder_decoder=self.is_encoder_decoder,
                tie_word_embeddings=self.tie_word_embeddings,
            )
            logger.info("`text_config` is `None`. initializing the `Pix2StructTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config["is_encoder_decoder"] = self.is_encoder_decoder
            self.text_config["tie_word_embeddings"] = self.tie_word_embeddings
            self.text_config = Pix2StructTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = Pix2StructVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Pix2StructVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Pix2StructVisionConfig(**self.vision_config)

        self.decoder_start_token_id = self.text_config.decoder_start_token_id
        self.pad_token_id = self.text_config.pad_token_id
        self.eos_token_id = self.text_config.eos_token_id

        self.text_config.initializer_range = self.initializer_range
        self.vision_config.initializer_range = self.initializer_range

        super().__post_init__(**kwargs)


__all__ = ["Pix2StructConfig", "Pix2StructTextConfig", "Pix2StructVisionConfig"]
