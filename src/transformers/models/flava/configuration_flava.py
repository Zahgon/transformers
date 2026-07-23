
from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/flava-full")
@strict
class FlavaImageConfig(PreTrainedConfig):

    model_type = "flava_image_model"
    base_config_key = "image_config"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    mask_token: bool = True
    vocab_size: int = 8192


@auto_docstring(checkpoint="facebook/flava-full")
@strict
class FlavaTextConfig(PreTrainedConfig):

    model_type = "flava_text_model"
    base_config_key = "text_config"

    vocab_size: int = 30522
    type_vocab_size: int = 2
    max_position_embeddings: int = 512
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    qkv_bias: bool = True


@auto_docstring(checkpoint="facebook/flava-full")
@strict
class FlavaMultimodalConfig(PreTrainedConfig):

    model_type = "flava_multimodal_model"
    base_config_key = "multimodal_config"

    hidden_size: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    qkv_bias: bool = True
    use_cls_token: bool = True


@auto_docstring(checkpoint="facebook/flava-full")
@strict
class FlavaImageCodebookConfig(PreTrainedConfig):

    num_groups: int = 4
    input_channels: int = 3
    num_blocks_per_group: int = 2
    hidden_size: int = 256
    vocab_size: int = 8192
    freeze: bool = True
    initializer_range: float = 0.02


@auto_docstring(checkpoint="facebook/flava-full")
@strict
class FlavaConfig(PreTrainedConfig):

    model_type = "flava"
    sub_configs = {
        "text_config": FlavaTextConfig,
        "image_config": FlavaImageConfig,
        "multimodal_config": FlavaMultimodalConfig,
        "image_codebook_config": FlavaImageCodebookConfig,
    }

    image_config: dict[str, Any] | PreTrainedConfig | None = None
    text_config: dict[str, Any] | PreTrainedConfig | None = None
    multimodal_config: dict[str, Any] | PreTrainedConfig | None = None
    image_codebook_config: dict[str, Any] | PreTrainedConfig | None = None
    hidden_size: int = 768
    layer_norm_eps: float = 1e-12
    projection_dim: int = 768
    init_codebook: bool = True
    logit_scale_init_value: float = 2.6592
    initializer_range: float = 0.02
    ce_ignore_index: int = -100
    mim_weight: float = 1.0
    mlm_weight: float = 1.0
    global_contrastive_weight: float = 1.0
    itm_weight: float = 1.0
    mmm_image_weight: float = 1.0
    mmm_text_weight: float = 1.0
    global_backprop_contrastive: bool = True
    skip_unmasked_multimodal_encoder: bool = True
    return_loss: bool = True
    tie_word_embeddings: bool = True
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `FlavaTextConfig` with default values.")
        elif isinstance(self.text_config, FlavaTextConfig):
            text_config = self.text_config.to_dict()
        else:
            text_config = self.text_config

        if self.image_config is None:
            image_config = {}
            logger.info("`image_config` is `None`. initializing the `FlavaImageConfig` with default values.")
        elif isinstance(self.image_config, FlavaImageConfig):
            image_config = self.image_config.to_dict()
        else:
            image_config = self.image_config

        if self.multimodal_config is None:
            multimodal_config = {}
            logger.info("`multimodal_config` is `None`. Initializing the `FlavaMultimodalConfig` with default values.")
        elif isinstance(self.multimodal_config, FlavaMultimodalConfig):
            multimodal_config = self.multimodal_config.to_dict()
        else:
            multimodal_config = self.multimodal_config

        if self.image_codebook_config is None:
            image_codebook_config = {}
            logger.info(
                "`image_codebook_config` is `None`. initializing the `FlavaImageCodebookConfig` with default values."
            )
        elif isinstance(self.image_codebook_config, FlavaImageCodebookConfig):
            image_codebook_config = self.image_codebook_config.to_dict()
        else:
            image_codebook_config = self.image_codebook_config

        text_config_dict = kwargs.pop("text_config_dict", None)
        image_config_dict = kwargs.pop("image_config_dict", None)
        multimodal_config_dict = kwargs.pop("multimodal_config_dict", None)
        image_codebook_config_dict = kwargs.pop("image_codebook_config_dict", None)

        if text_config_dict is not None:
            _text_config_dict = FlavaTextConfig(**text_config_dict).to_dict()

            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key != "transformers_version":
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `FlavaTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            text_config.update(_text_config_dict)

        if image_config_dict is not None:
            _image_config_dict = FlavaImageConfig(**image_config_dict).to_dict()
            if "id2label" in _image_config_dict:
                _image_config_dict["id2label"] = {
                    str(key): value for key, value in _image_config_dict["id2label"].items()
                }

            for key, value in _image_config_dict.items():
                if key in image_config and value != image_config[key] and key != "transformers_version":
                    if key in image_config_dict:
                        message = (
                            f"`{key}` is found in both `image_config_dict` and `image_config` but with different "
                            f'values. The value `image_config_dict["{key}"]` will be used instead.'
                        )
                    else:
                        message = (
                            f"`image_config_dict` is provided which will be used to initialize `FlavaImageConfig`. "
                            f'The value `image_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            image_config.update(_image_config_dict)

        if multimodal_config_dict is not None:
            _multimodal_config_dict = FlavaMultimodalConfig(**multimodal_config_dict).to_dict()

            for key, value in _multimodal_config_dict.items():
                if key in multimodal_config and value != multimodal_config[key] and key != "transformers_version":
                    if key in multimodal_config_dict:
                        message = (
                            f"`{key}` is found in both `multimodal_config_dict` and `multimodal_config` but with "
                            f'different values. The value `multimodal_config_dict["{key}"]` will be used instead.'
                        )
                    else:
                        message = (
                            f"`multimodal_config_dict` is provided which will be used to initialize "
                            f'`FlavaMultimodalConfig`. The value `multimodal_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            multimodal_config.update(_multimodal_config_dict)

        if image_codebook_config_dict is not None:
            _image_codebook_config_dict = FlavaImageCodebookConfig(**image_codebook_config_dict).to_dict()

            for key, value in _image_codebook_config_dict.items():
                if (
                    key in image_codebook_config
                    and value != image_codebook_config[key]
                    and key != "transformers_version"
                ):
                    if key in image_codebook_config_dict:
                        message = (
                            f"`{key}` is found in both `image_codebook_config_dict` and `image_codebook_config` but "
                            f'with different values. The value `image_codebook_config_dict["{key}"]` will be used '
                            "instead."
                        )
                    else:
                        message = (
                            f"`image_codebook_config_dict` is provided which will be used to initialize "
                            f'`FlavaImageCodebookConfig`. The value `image_codebook_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            image_codebook_config.update(_image_codebook_config_dict)

        self.text_config = FlavaTextConfig(**text_config)
        self.image_config = FlavaImageConfig(**image_config)
        self.multimodal_config = FlavaMultimodalConfig(**multimodal_config)
        self.image_codebook_config = FlavaImageCodebookConfig(**image_codebook_config)

        super().__post_init__(**kwargs)


__all__ = ["FlavaConfig", "FlavaImageCodebookConfig", "FlavaImageConfig", "FlavaMultimodalConfig", "FlavaTextConfig"]
