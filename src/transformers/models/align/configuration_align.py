
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="kakaobrain/align-base")
@strict
class AlignTextConfig(PreTrainedConfig):

    model_type = "align_text_model"
    base_config_key = "text_config"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None


@auto_docstring(checkpoint="kakaobrain/align-base")
@strict
class AlignVisionConfig(PreTrainedConfig):

    model_type = "align_vision_model"
    base_config_key = "vision_config"

    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 600
    width_coefficient: float = 2.0
    depth_coefficient: float = 3.1
    depth_divisor: int = 8
    kernel_sizes: list[int] | tuple[int, ...] = (3, 3, 5, 3, 5, 5, 3)
    in_channels: list[int] | tuple[int, ...] = (32, 16, 24, 40, 80, 112, 192)
    out_channels: list[int] | tuple[int, ...] = (16, 24, 40, 80, 112, 192, 320)
    depthwise_padding: list | tuple[int, ...] = ()
    strides: list[int] | tuple[int, ...] = (1, 2, 2, 2, 1, 2, 1)
    num_block_repeats: list[int] | tuple[int, ...] = (1, 2, 2, 3, 3, 4, 1)
    expand_ratios: list[int] | tuple[int, ...] = (1, 6, 6, 6, 6, 6, 6)
    squeeze_expansion_ratio: float = 0.25
    hidden_act: str = "swish"
    hidden_dim: int = 2560
    pooling_type: str = "mean"
    initializer_range: float = 0.02
    batch_norm_eps: float = 0.001
    batch_norm_momentum: float = 0.99
    drop_connect_rate: float | int = 0.2

    def __post_init__(self, **kwargs):
        self.num_hidden_layers = sum(self.num_block_repeats) * 4
        for attr in [
            "kernel_sizes",
            "in_channels",
            "out_channels",
            "depthwise_padding",
            "strides",
            "num_block_repeats",
            "expand_ratios",
        ]:
            setattr(self, attr, list(getattr(self, attr)))
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="kakaobrain/align-base")
@strict
class AlignConfig(PreTrainedConfig):

    model_type = "align"
    sub_configs = {"text_config": AlignTextConfig, "vision_config": AlignVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 640
    temperature_init_value: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = AlignTextConfig()
            logger.info("`text_config` is `None`. Initializing the `AlignTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = AlignTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = AlignVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `AlignVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = AlignVisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["AlignTextConfig", "AlignVisionConfig", "AlignConfig"]
