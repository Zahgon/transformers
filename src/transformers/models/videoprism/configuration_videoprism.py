

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/videoprism-base-f16r288")
@strict
class VideoPrismVisionConfig(PreTrainedConfig):

    model_type = "videoprism_vision_model"
    image_size: int | list[int] | tuple[int, int] = 288
    num_frames: int = 16
    tubelet_size: list[int] | tuple[int, ...] = (1, 18, 18)
    num_channels: int = 3
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_python"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-06
    qkv_bias: bool = True
    base_config_key = "vision_config"
    num_spatial_layers: int = 12
    num_temporal_layers: int = 4
    attn_logit_softcapping: float = 50.0
    num_auxiliary_layers: int = 2
    apply_l2norm: bool = True


@auto_docstring(checkpoint="google/videoprism-lvt-base-f16r288")
@strict
class VideoPrismTextConfig(PreTrainedConfig):

    model_type = "videoprism_text_model"
    base_config_key = "text_config"

    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 64

    hidden_act: str = "relu"
    layer_norm_eps: float = 1e-6
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    attention_probs_dropout_prob: float | int = 0.0
    apply_l2norm: bool = True
    qkv_bias: bool = True
    hidden_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    attn_logit_softcapping: float = 50.0


@auto_docstring(checkpoint="google/videoprism-lvt-base-f16r288")
@strict
class VideoPrismConfig(PreTrainedConfig):

    model_type = "videoprism"
    sub_configs = {"text_config": VideoPrismTextConfig, "vision_config": VideoPrismVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = VideoPrismTextConfig()
            logger.info("`text_config` is `None`. Initializing the `VideoPrismTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = VideoPrismTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = VideoPrismVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `VideoPrismVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = VideoPrismVisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["VideoPrismVisionConfig", "VideoPrismTextConfig", "VideoPrismConfig"]
