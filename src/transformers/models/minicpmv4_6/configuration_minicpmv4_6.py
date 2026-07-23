

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4.6")
@strict
class MiniCPMV4_6VisionConfig(PreTrainedConfig):

    model_type = "minicpmv4_6_vision"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    insert_layer_id: int = 6
    window_kernel_size: tuple[int, int] | list[int] = (2, 2)

    @property
    def window_hidden_size(self) -> int:
        pass

    @property
    def window_intermediate_size(self) -> int:
        pass


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4.6")
@strict
class MiniCPMV4_6Config(PreTrainedConfig):

    model_type = "minicpmv4_6"
    sub_configs = {"text_config": AutoConfig, "vision_config": MiniCPMV4_6VisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    insert_layer_id: int = 6
    image_size: int = 448
    drop_vision_last_layer: bool = False
    image_token_id: int | None = None
    video_token_id: int | None = None
    tie_word_embeddings: bool = False
    downsample_mode: str = "16x"
    merge_kernel_size: tuple[int, int] | list[int] = (2, 2)
    merger_times: int = 1

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config.pop("model_type", None)
            self.vision_config = MiniCPMV4_6VisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = MiniCPMV4_6VisionConfig()

        self.vision_config.insert_layer_id = self.insert_layer_id
        self.patch_size = self.vision_config.patch_size

        if isinstance(self.text_config, dict):
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3_5_text"]()

        super().__post_init__(**kwargs)


__all__ = ["MiniCPMV4_6Config", "MiniCPMV4_6VisionConfig"]
