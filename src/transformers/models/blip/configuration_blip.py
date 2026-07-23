
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Salesforce/blip-vqa-base")
@strict
class BlipTextConfig(PreTrainedConfig):

    model_type = "blip_text_model"
    base_config_key = "text_config"

    vocab_size: int = 30524
    hidden_size: int = 768
    encoder_hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 512
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    bos_token_id: int | None = 30522
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = 0
    sep_token_id: int | None = 102
    is_decoder: bool = True
    use_cache: bool = True
    tie_word_embeddings: bool = True
    label_smoothing: float = 0.0


@auto_docstring(checkpoint="Salesforce/blip-vqa-base")
@strict
class BlipVisionConfig(PreTrainedConfig):

    model_type = "blip_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    image_size: int | list[int] | tuple[int, int] = 384
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10


@auto_docstring(checkpoint="Salesforce/blip-vqa-base")
@strict
class BlipConfig(PreTrainedConfig):

    model_type = "blip"
    sub_configs = {"text_config": BlipTextConfig, "vision_config": BlipVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592
    image_text_hidden_size: int = 256
    label_smoothing: float = 0.0
    tie_word_embeddings: bool = True
    initializer_factor: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = BlipTextConfig()
            logger.info("`text_config` is `None`. Initializing the `BlipTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = BlipTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = BlipVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `BlipVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = BlipVisionConfig(**self.vision_config)

        self.text_config.encoder_hidden_size = self.vision_config.hidden_size

        super().__post_init__(**kwargs)


__all__ = ["BlipConfig", "BlipTextConfig", "BlipVisionConfig"]
