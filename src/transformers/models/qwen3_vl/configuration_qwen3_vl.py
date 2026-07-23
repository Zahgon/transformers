from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="Qwen/Qwen3-VL-4B-Instruct")
@strict
class Qwen3VLVisionConfig(PreTrainedConfig):

    model_type = "qwen3_vl_vision"
    base_config_key = "vision_config"

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] | tuple[int, ...] = (8, 16, 24)
    initializer_range: float = 0.02


@auto_docstring(checkpoint="Qwen/Qwen3-VL-4B-Instruct")
@strict
class Qwen3VLTextConfig(PreTrainedConfig):

    model_type = "qwen3_vl_text"
    base_config_key = "text_config"
    default_theta = 500000.0
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen3-VL-4B-Instruct")
@strict
class Qwen3VLConfig(PreTrainedConfig):

    model_type = "qwen3_vl"
    sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": Qwen3VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            if self.vision_config.get("model_type") == "qwen3_vl":
                self.vision_config["model_type"] = "qwen3_vl_vision"
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        super().__post_init__(**kwargs)


__all__ = ["Qwen3VLConfig", "Qwen3VLTextConfig", "Qwen3VLVisionConfig"]
