

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="baidu/Qianfan-OCR")
@strict
class QianfanOCRVisionConfig(PreTrainedConfig):

    model_type = "qianfan_ocr_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16

    attention_bias: bool = True
    use_qk_norm: bool = False
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_dropout: float | int = 0.0
    projection_dropout: float | int = 0.0
    initializer_range: float = 0.02
    norm_type: str = "layer_norm"
    layer_norm_eps: float = 1e-06
    image_size: int | list[int] | tuple[int, ...] = (448, 448)
    patch_size: int | list[int] | tuple[int, ...] = (14, 14)
    num_channels: int = 3
    use_mask_token: bool = False
    use_absolute_position_embeddings: bool = True
    layer_scale_init_value: float = 0.1
    use_mean_pooling: bool = True
    drop_path_rate: float = 0.1

    def __post_init__(self, **kwargs):
        self.image_size = (
            self.image_size if isinstance(self.image_size, (list, tuple)) else (self.image_size, self.image_size)
        )
        self.patch_size = (
            self.patch_size if isinstance(self.patch_size, (list, tuple)) else (self.patch_size, self.patch_size)
        )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="baidu/Qianfan-OCR")
@strict
class QianfanOCRConfig(PreTrainedConfig):

    model_type = "qianfan_ocr"
    sub_configs = {"text_config": AutoConfig, "vision_config": QianfanOCRVisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151667
    image_seq_length: int = 256
    downsample_ratio: float = 0.5
    projector_hidden_act: str = "gelu"
    vision_feature_layer: int | list[int] = -1
    vision_feature_select_strategy: str = "default"

    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = QianfanOCRVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = QianfanOCRVisionConfig()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()

        super().__post_init__(**kwargs)


__all__ = ["QianfanOCRVisionConfig", "QianfanOCRConfig"]
