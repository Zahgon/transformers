
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Salesforce/instructblip-flan-t5-xl")
@strict
class InstructBlipVisionConfig(PreTrainedConfig):

    model_type = "instructblip_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 1408
    intermediate_size: int = 6144
    num_hidden_layers: int = 39
    num_attention_heads: int = 16
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10
    qkv_bias: bool = True


@auto_docstring(checkpoint="Salesforce/instructblip-flan-t5-xl")
@strict
class InstructBlipQFormerConfig(PreTrainedConfig):

    model_type = "instructblip_qformer"
    base_config_key = "qformer_config"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    cross_attention_frequency: int = 2
    encoder_hidden_size: int = 1408


@auto_docstring(checkpoint="Salesforce/instructblip-flan-t5-xl")
@strict
class InstructBlipConfig(PreTrainedConfig):

    model_type = "instructblip"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {
        "text_config": AutoConfig,
        "qformer_config": InstructBlipQFormerConfig,
        "vision_config": InstructBlipVisionConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    qformer_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    num_query_tokens: int = 32
    image_token_index: int | None = None
    initializer_factor: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = CONFIG_MAPPING["opt"]()
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")
        elif isinstance(self.text_config, dict):
            text_model_type = self.text_config.get("model_type", "opt")
            self.text_config = CONFIG_MAPPING[text_model_type](**self.text_config)

        if self.qformer_config is None:
            self.qformer_config = InstructBlipQFormerConfig()
            logger.info("qformer_config is None. Initializing the InstructBlipQFormerConfig with default values.")
        elif isinstance(self.qformer_config, dict):
            self.qformer_config = InstructBlipQFormerConfig(**self.qformer_config)

        if self.vision_config is None:
            self.vision_config = InstructBlipVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `InstructBlipVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = InstructBlipVisionConfig(**self.vision_config)

        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        super().__post_init__(**kwargs)


__all__ = ["InstructBlipConfig", "InstructBlipQFormerConfig", "InstructBlipVisionConfig"]
