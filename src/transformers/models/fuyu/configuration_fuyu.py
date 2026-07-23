
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="adept/fuyu-8b")
@strict
class FuyuConfig(PreTrainedConfig):

    model_type = "fuyu"
    sub_configs = {"text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 25000.0

    vocab_size: int = 262144
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 36
    num_attention_heads: int = 64
    hidden_act: str = "relu2"
    max_position_embeddings: int = 16384
    image_size: int | None = 300
    patch_size: int | None = 30
    num_channels: int | None = 3
    initializer_range: float = 0.02
    layer_norm_eps: float | None = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    qk_layernorm: bool | None = True
    hidden_dropout: float | int | None = 0.0
    attention_dropout: float | int | None = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    image_token_id: int | None = 71011
    text_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            text_config = {
                "vocab_size": self.vocab_size,
                "max_position_embeddings": self.max_position_embeddings,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "hidden_act": self.hidden_act,
                "initializer_range": self.initializer_range,
                "layer_norm_eps": self.layer_norm_eps,
                "use_cache": self.use_cache,
                "rope_parameters": self.rope_parameters,
                "qk_layernorm": self.qk_layernorm,
                "hidden_dropout": self.hidden_dropout,
                "attention_dropout": self.attention_dropout,
                "pad_token_id": self.pad_token_id,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
            }
            logger.info("text_config is None. initializing the text model with default values.")
            self.text_config = CONFIG_MAPPING["persimmon"](**text_config)
        elif isinstance(self.text_config, dict):
            text_model_type = self.text_config.get("model_type", "persimmon")
            self.text_config = CONFIG_MAPPING[text_model_type](**self.text_config)

        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        super().__post_init__(**kwargs)


__all__ = ["FuyuConfig"]
