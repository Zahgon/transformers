
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="meta-llama/Llama-3.2-11B-Vision")
@strict
class MllamaVisionConfig(PreTrainedConfig):

    model_type = "mllama_vision_model"
    base_config_key = "vision_config"
    attribute_map = {"num_attention_heads": "attention_heads"}

    hidden_size: int = 1280
    hidden_act: str = "gelu"
    num_hidden_layers: int = 32
    num_global_layers: int = 8
    attention_heads: int = 16
    num_channels: int = 3
    intermediate_size: int = 5120
    vision_output_dim: int = 7680
    image_size: int | list[int] | tuple[int, int] = 448
    patch_size: int | list[int] | tuple[int, int] = 14
    norm_eps: float = 1e-5
    max_num_tiles: int = 4
    intermediate_layers_indices: list[int] | None = None
    supported_aspect_ratios: list[list[int]] | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.supported_aspect_ratios is None:
            self.supported_aspect_ratios = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]

        if self.intermediate_layers_indices is None:
            self.intermediate_layers_indices = [3, 7, 15, 23, 30]
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def max_aspect_ratio_id(self) -> int:
        pass


@auto_docstring(checkpoint="meta-llama/Llama-3.2-11B-Vision")
@strict
class MllamaTextConfig(PreTrainedConfig):

    model_type = "mllama_text_model"
    base_config_key = "text_config"
    default_theta = 500000.0

    vocab_size: int = 128256
    hidden_size: int = 4096
    hidden_act: str = "silu"
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14_336
    rope_parameters: dict | None = None
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False
    cross_attention_layers: list[int] | None = None
    dropout: float | int = 0.0
    bos_token_id: int = 128000
    eos_token_id: int | list[int] | None = 128001
    pad_token_id: int | None = 128004

    def __post_init__(self, **kwargs):
        if self.cross_attention_layers is None:
            self.cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="meta-llama/Llama-3.2-11B-Vision")
@strict
class MllamaConfig(PreTrainedConfig):

    model_type = "mllama"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": MllamaTextConfig, "vision_config": MllamaVisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 128256

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = MllamaVisionConfig()
            logger.info("vision_config is None, using default mllama vision config")
        elif isinstance(self.vision_config, dict):
            self.vision_config = MllamaVisionConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = MllamaTextConfig()
            logger.info("text_config is None, using default mllama text config")
        elif isinstance(self.text_config, dict):
            self.text_config = MllamaTextConfig(**self.text_config)

        super().__post_init__(**kwargs)


__all__ = ["MllamaConfig", "MllamaTextConfig", "MllamaVisionConfig"]
