
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="suno/bark")
@strict
class BarkSubModelConfig(PreTrainedConfig):

    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "vocab_size": "input_vocab_size",
        "window_size": "block_size",
    }

    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    num_layers: int = 12
    num_heads: int = 12
    hidden_size: int = 768
    dropout: float | int = 0.0
    bias: bool = True
    initializer_range: float = 0.02
    use_cache: bool = True


@auto_docstring(checkpoint="suno/bark")
@strict
class BarkSemanticConfig(BarkSubModelConfig):

    model_type = "semantic"
    base_config_key = "semantic_config"


@auto_docstring(checkpoint="suno/bark")
@strict
class BarkCoarseConfig(BarkSubModelConfig):

    model_type = "coarse_acoustics"
    base_config_key = "coarse_acoustics_config"


@auto_docstring(checkpoint="suno/bark")
@strict
class BarkFineConfig(BarkSubModelConfig):

    model_type = "fine_acoustics"
    base_config_key = "fine_acoustics_config"

    tie_word_embeddings: bool = True
    n_codes_total: int = 8
    n_codes_given: int = 1


@auto_docstring(checkpoint="suno/bark")
@strict
class BarkConfig(PreTrainedConfig):

    model_type = "bark"
    sub_configs = {
        "semantic_config": BarkSemanticConfig,
        "coarse_acoustics_config": BarkCoarseConfig,
        "fine_acoustics_config": BarkFineConfig,
        "codec_config": AutoConfig,
    }
    semantic_config: dict | PreTrainedConfig | None = None
    coarse_acoustics_config: dict | PreTrainedConfig | None = None
    fine_acoustics_config: dict | PreTrainedConfig | None = None
    codec_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.semantic_config is None:
            self.semantic_config = BarkSemanticConfig()
            logger.info("`semantic_config` is `None`. Initializing the `BarkSemanticConfig` with default values.")
        elif isinstance(self.semantic_config, dict):
            self.semantic_config = BarkSemanticConfig(**self.semantic_config)

        if self.coarse_acoustics_config is None:
            self.coarse_acoustics_config = BarkCoarseConfig()
            logger.info(
                "`coarse_acoustics_config` is `None`. Initializing the `BarkCoarseConfig` with default values."
            )
        elif isinstance(self.coarse_acoustics_config, dict):
            self.coarse_acoustics_config = BarkCoarseConfig(**self.coarse_acoustics_config)

        if self.fine_acoustics_config is None:
            self.fine_acoustics_config = BarkFineConfig()
            logger.info("`fine_acoustics_config` is `None`. Initializing the `BarkFineConfig` with default values.")
        elif isinstance(self.fine_acoustics_config, dict):
            self.fine_acoustics_config = BarkFineConfig(**self.fine_acoustics_config)

        if self.codec_config is None:
            self.codec_config = CONFIG_MAPPING["encodec"]()
            logger.info("`codec_config` is `None`. Initializing the `codec_config` with default values.")
        elif isinstance(self.codec_config, dict):
            codec_model_type = self.codec_config.get("model_type", "encodec")
            self.codec_config = CONFIG_MAPPING[codec_model_type](**self.codec_config)

        super().__post_init__(**kwargs)


__all__ = ["BarkCoarseConfig", "BarkConfig", "BarkFineConfig", "BarkSemanticConfig"]
