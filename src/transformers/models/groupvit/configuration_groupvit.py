
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/groupvit-gcc-yfcc")
@strict
class GroupViTTextConfig(PreTrainedConfig):

    model_type = "groupvit_text_model"
    base_config_key = "text_config"

    vocab_size: int = 49408
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 49406
    eos_token_id: int | list[int] | None = 49407


@auto_docstring(checkpoint="nvidia/groupvit-gcc-yfcc")
@strict
class GroupViTVisionConfig(PreTrainedConfig):

    model_type = "groupvit_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    depths: list[int] | tuple[int, ...] = (6, 3, 3)
    num_group_tokens: list[int] | tuple[int, ...] = (64, 8, 0)
    num_output_groups: list[int] | tuple[int, ...] = (64, 8, 8)
    num_attention_heads: int = 6
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    assign_eps: float = 1.0
    assign_mlp_ratio: list[float | int] | tuple[float | int, ...] = (0.5, 4)

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="nvidia/groupvit-gcc-yfcc")
@strict
class GroupViTConfig(PreTrainedConfig):

    model_type = "groupvit"
    sub_configs = {"text_config": GroupViTTextConfig, "vision_config": GroupViTVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 256
    projection_intermediate_dim: int = 4096
    logit_scale_init_value: float = 2.6592
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    output_segmentation: bool = False

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `GroupViTTextConfig` with default values.")
        elif isinstance(self.text_config, GroupViTTextConfig):
            text_config = self.text_config.to_dict()
        else:
            text_config = self.text_config

        if self.vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `GroupViTVisionConfig` with default values.")
        elif isinstance(self.vision_config, GroupViTVisionConfig):
            vision_config = self.vision_config.to_dict()
        else:
            vision_config = self.vision_config

        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        if text_config_dict is not None:
            _text_config_dict = GroupViTTextConfig(**text_config_dict).to_dict()

            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key != "transformers_version":
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `GroupViTTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            _vision_config_dict = GroupViTVisionConfig(**vision_config_dict).to_dict()
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
                            f"`vision_config_dict` is provided which will be used to initialize `GroupViTVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            vision_config.update(_vision_config_dict)

        self.text_config = GroupViTTextConfig(**text_config)
        self.vision_config = GroupViTVisionConfig(**vision_config)

        super().__post_init__(**kwargs)


__all__ = ["GroupViTConfig", "GroupViTTextConfig", "GroupViTVisionConfig"]
