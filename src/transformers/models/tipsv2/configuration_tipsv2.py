
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/tipsv2-b14")
@strict
class Tipsv2VisionConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "tipsv2_vision_model"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int | float = 4  # float required for so400m14 checkpoint
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int | list[int] | tuple[int, int] = 448

    patch_size: int | list[int] | tuple[int, int] = 14
    num_channels: int = 3
    qkv_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float | int = 0.0
    use_swiglu_ffn: bool = False
    num_register_tokens: int = 1
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    apply_layernorm: bool = True
    reshape_hidden_states: bool = True
    base_config_key = "vision_config"

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="google/tipsv2-b14")
@strict
class Tipsv2TextConfig(PreTrainedConfig):

    model_type = "tipsv2_text_model"
    base_config_key = "text_config"

    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 64

    hidden_act: str = "relu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    initializer_range: float = 0.02
    scale_sqrt_depth: bool = True
    pooling_epsilon: float = 1e-8

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="google/tipsv2-b14")
@strict
class Tipsv2Config(PreTrainedConfig):

    model_type = "tipsv2"
    sub_configs = {"text_config": Tipsv2TextConfig, "vision_config": Tipsv2VisionConfig}

    text_config: dict | Tipsv2TextConfig | None = None
    vision_config: dict | Tipsv2VisionConfig | None = None
    temperature_init_value: float = 0.005065968260169029

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["Tipsv2Config", "Tipsv2TextConfig", "Tipsv2VisionConfig"]
