from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="BAAI/AltCLIP")
@strict
class AltCLIPTextConfig(PreTrainedConfig):

    model_type = "altclip_text_model"
    base_config_key = "text_config"

    vocab_size: int = 250002
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    max_position_embeddings: int = 514
    hidden_act: str = "gelu"
    layer_norm_eps: float | None = 1e-5
    initializer_range: float = 0.02
    initializer_factor: float = 0.02
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | None = 2
    hidden_dropout_prob: int | float = 0.1
    attention_probs_dropout_prob: int | float = 0
    type_vocab_size: int = 1
    project_dim: int = 768

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="BAAI/AltCLIP")
@strict
class AltCLIPVisionConfig(PreTrainedConfig):

    model_type = "altclip_vision_model"
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


@auto_docstring(checkpoint="BAAI/AltCLIP")
@strict
class AltCLIPConfig(PreTrainedConfig):

    model_type = "altclip"
    sub_configs = {"text_config": AltCLIPTextConfig, "vision_config": AltCLIPVisionConfig}

    text_config: dict | AltCLIPTextConfig | None = None
    vision_config: dict | AltCLIPVisionConfig | None = None

    projection_dim: int = 768
    logit_scale_init_value: float | int | None = 2.6592
    initializer_factor: float | None = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `AltCLIPTextConfig` with default values.")
        elif isinstance(self.text_config, AltCLIPTextConfig):
            text_config = self.text_config.to_dict()
        else:
            text_config = self.text_config

        if self.vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `AltCLIPVisionConfig` with default values.")
        elif isinstance(self.vision_config, AltCLIPVisionConfig):
            vision_config = self.vision_config.to_dict()
        else:
            vision_config = self.vision_config

        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        if text_config_dict is not None:
            _text_config_dict = AltCLIPTextConfig(**text_config_dict).to_dict()

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
            _vision_config_dict = AltCLIPVisionConfig(**vision_config_dict).to_dict()
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

        self.text_config = AltCLIPTextConfig(**text_config)
        self.vision_config = AltCLIPVisionConfig(**vision_config)

        super().__post_init__(**kwargs)


__all__ = ["AltCLIPTextConfig", "AltCLIPVisionConfig", "AltCLIPConfig"]
