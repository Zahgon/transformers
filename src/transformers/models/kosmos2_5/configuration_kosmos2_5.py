
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/kosmos-2.5")
@strict
class Kosmos2_5TextConfig(PreTrainedConfig):
    model_type = "kosmos_2_5_text_model"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "embed_dim",
        "num_hidden_layers": "layers",
    }

    vocab_size: int = 108481
    max_position_embeddings: int = 4096
    embed_dim: int = 1536
    layers: int = 24
    ffn_dim: int = 6144
    attention_heads: int = 16
    activation_function: str = "gelu"
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    layerdrop: float | int = 0.0
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02
    scale_embedding: bool = True
    use_cache: bool = True
    tie_word_embeddings: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2


@auto_docstring(checkpoint="microsoft/kosmos-2.5")
@strict
class Kosmos2_5VisionConfig(PreTrainedConfig):

    model_type = "kosmos_2_5_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 1536
    patch_embed_hidden_size: int = 768
    intermediate_size: int = 3968
    head_dim: int = 64
    num_hidden_layers: int = 18
    num_attention_heads: int = 24
    dense_act_fn: str = "gelu_new"
    layer_norm_eps: float = 1e-6
    dropout_rate: float | int = 0.0
    attention_dropout: float | int = 0.0
    max_num_patches: int = 4096
    initializer_factor: float = 1.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="microsoft/kosmos-2.5")
@strict
class Kosmos2_5Config(PreTrainedConfig):

    model_type = "kosmos-2.5"
    sub_configs = {"text_config": Kosmos2_5TextConfig, "vision_config": Kosmos2_5VisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    latent_query_num: int = 2048
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Kosmos2_5TextConfig()
            logger.info("`text_config` is `None`. initializing the `Kosmos2_5TextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = Kosmos2_5TextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = Kosmos2_5VisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Kosmos2_5VisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Kosmos2_5VisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["Kosmos2_5Config", "Kosmos2_5TextConfig", "Kosmos2_5VisionConfig"]
