
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/kosmos-2-patch14-224")
@strict
class Kosmos2TextConfig(PreTrainedConfig):
    model_type = "kosmos_2_text_model"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "embed_dim",
        "num_hidden_layers": "layers",
    }

    vocab_size: int = 65037
    max_position_embeddings: int = 2048
    embed_dim: int = 2048
    layers: int = 24
    ffn_dim: int = 8192
    attention_heads: int = 32
    activation_function: str = "gelu"
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
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
    add_cross_attention: bool = False


@auto_docstring(checkpoint="microsoft/kosmos-2-patch14-224")
@strict
class Kosmos2VisionConfig(PreTrainedConfig):
    model_type = "kosmos_2_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 14
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0


@auto_docstring(checkpoint="microsoft/kosmos-2-patch14-224")
@strict
class Kosmos2Config(PreTrainedConfig):

    model_type = "kosmos-2"
    sub_configs = {"text_config": Kosmos2TextConfig, "vision_config": Kosmos2VisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    latent_query_num: int = 64
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Kosmos2TextConfig()
            logger.info("`text_config` is `None`. initializing the `Kosmos2TextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = Kosmos2TextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = Kosmos2VisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Kosmos2VisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Kosmos2VisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["Kosmos2Config", "Kosmos2TextConfig", "Kosmos2VisionConfig"]
