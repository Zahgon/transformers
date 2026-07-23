from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="OFA-Sys/chinese-clip-vit-base-patch16")
@strict
class ChineseCLIPTextConfig(PreTrainedConfig):

    model_type = "chinese_clip_text_model"
    base_config_key = "text_config"

    vocab_size: int = 30522
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    initializer_factor: float | None = 1.0
    pad_token_id: int | None = 0
    bos_token_id: int | None = 0
    eos_token_id: int | None = None
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    type_vocab_size: int = 2

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="OFA-Sys/chinese-clip-vit-base-patch16")
@strict
class ChineseCLIPVisionConfig(PreTrainedConfig):

    model_type = "chinese_clip_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] | None = 224
    patch_size: int | list[int] | tuple[int, int] | None = 32
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: int | float | None = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="OFA-Sys/chinese-clip-vit-base-patch16")
@strict
class ChineseCLIPConfig(PreTrainedConfig):

    model_type = "chinese_clip"
    sub_configs = {"text_config": ChineseCLIPTextConfig, "vision_config": ChineseCLIPVisionConfig}

    text_config: dict | ChineseCLIPTextConfig | None = None
    vision_config: dict | ChineseCLIPVisionConfig | None = None
    projection_dim: int | None = 512
    logit_scale_init_value: float | int | None = 2.6592
    initializer_factor: float | None = 1.0

    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `ChineseCLIPTextConfig` with default values.")
        elif isinstance(self.text_config, ChineseCLIPTextConfig):
            text_config = self.text_config.to_dict()
        else:
            text_config = self.text_config

        if self.vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `ChineseCLIPVisionConfig` with default values.")
        elif isinstance(self.vision_config, ChineseCLIPVisionConfig):
            vision_config = self.vision_config.to_dict()
        else:
            vision_config = self.vision_config

        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        if text_config_dict is not None:
            _text_config_dict = ChineseCLIPTextConfig(**text_config_dict).to_dict()

            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key != "transformers_version":
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            _vision_config_dict = ChineseCLIPVisionConfig(**vision_config_dict).to_dict()
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key != "transformers_version":
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `CLIPVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            vision_config.update(_vision_config_dict)

        self.text_config = ChineseCLIPTextConfig(**text_config)
        self.vision_config = ChineseCLIPVisionConfig(**vision_config)

        super().__post_init__(**kwargs)


__all__ = ["ChineseCLIPConfig", "ChineseCLIPTextConfig", "ChineseCLIPVisionConfig"]
