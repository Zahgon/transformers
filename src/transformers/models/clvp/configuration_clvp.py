
import os

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="susnato/clvp_dev")
@strict
class ClvpEncoderConfig(PreTrainedConfig):

    model_type = "clvp_encoder"
    base_config_key = ["text_config", "speech_config"]

    vocab_size: int = 256
    hidden_size: int = 768
    intermediate_size: int = 1536
    projection_dim: int = 768
    num_hidden_layers: int = 20
    num_attention_heads: int = 12
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.1
    dropout: float | int = 0.1
    use_rotary_embedding: bool = True
    use_attention_bias: bool = False
    summary_type: str = "mean"
    initializer_factor: float = 1.0
    bos_token_id: int | None = 255
    eos_token_id: int | list[int] | None = 0
    pad_token_id: int | None = None

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, config_type: str = "text_config", **kwargs
    ):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_type not in cls.base_config_key:
            raise ValueError(
                f"We can only load either 'text_config' or 'speech_config' but you are trying to load{config_type}"
            )

        if config_dict.get("model_type") == "clvp":
            config_dict = config_dict[config_type]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


@auto_docstring(checkpoint="susnato/clvp_dev")
@strict
class ClvpDecoderConfig(PreTrainedConfig):

    model_type = "clvp_decoder"
    base_config_key = "decoder_config"

    vocab_size: int = 8194
    max_position_embeddings: int = 608
    max_text_tokens: int = 404
    hidden_size: int = 1024
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    n_inner: int | None = None
    num_mel_attn_blocks: int = 6
    activation_function: str = "gelu_new"
    resid_pdrop: float | int = 0.1
    embd_pdrop: float | int = 0.1
    attention_dropout: float | int = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    summary_type: str = "cls_index"
    summary_use_proj: bool = True
    summary_activation: str | None = None
    summary_proj_to_labels: bool = True
    summary_first_dropout: float | int = 0.1
    use_cache: bool = True
    bos_token_id: int | None = 8192
    eos_token_id: int | list[int] | None = 8193
    pad_token_id: int | None = None
    feature_size: int = 80
    use_attention_bias: bool = True
    initializer_factor: float = 1.0
    decoder_fixing_codes: list[int] | tuple[int, ...] = (83, 45, 45, 248)
    add_cross_attention: bool = False


@auto_docstring(checkpoint="susnato/clvp_dev")
@strict
class ClvpConfig(PreTrainedConfig):

    model_type = "clvp"
    sub_configs = {
        "text_config": ClvpEncoderConfig,
        "speech_config": ClvpEncoderConfig,
        "decoder_config": ClvpDecoderConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    speech_config: dict | PreTrainedConfig | None = None
    decoder_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 768
    logit_scale_init_value: float = 2.6592
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = ClvpEncoderConfig()
            logger.info("`text_config` is `None`. initializing the `ClvpEncoderConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = ClvpEncoderConfig(**self.text_config)

        if self.speech_config is None:
            self.speech_config = ClvpEncoderConfig()
            logger.info("`speech_config` is `None`. initializing the `ClvpEncoderConfig` with default values.")
        elif isinstance(self.speech_config, dict):
            self.speech_config = ClvpEncoderConfig(**self.speech_config)

        if self.decoder_config is None:
            self.decoder_config = ClvpDecoderConfig()
            logger.info("`image_config` is `None`. initializing the `ClvpDecoderConfig` with default values.")
        elif isinstance(self.decoder_config, dict):
            self.decoder_config = ClvpDecoderConfig(**self.decoder_config)

        super().__post_init__(**kwargs)


__all__ = ["ClvpConfig", "ClvpDecoderConfig", "ClvpEncoderConfig"]
