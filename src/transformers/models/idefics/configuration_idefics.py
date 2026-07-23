
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="HuggingFaceM4/idefics-9b")
@strict
class IdeficsVisionConfig(PreTrainedConfig):
    model_type = "idefics_vision"
    attribute_map = {"hidden_size": "embed_dim"}

    embed_dim: int = 768
    image_size: int | list[int] | tuple[int, int] = 224
    intermediate_size: int = 5120
    patch_size: int | list[int] | tuple[int, int] = 14
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0


@auto_docstring(checkpoint="HuggingFaceM4/idefics-9b")
@strict
class IdeficsPerceiverConfig(PreTrainedConfig):

    model_type = "idefics_perciever"

    use_resampler: bool = False
    resampler_n_latents: int = 64
    resampler_depth: int = 6
    resampler_n_heads: int = 16
    resampler_head_dim: int = 96
    qk_layer_norms_perceiver: bool = False


@auto_docstring(checkpoint="HuggingFaceM4/idefics-9b")
@strict
class IdeficsConfig(PreTrainedConfig):

    model_type = "idefics"
    sub_configs = {"perceiver_config": IdeficsPerceiverConfig, "vision_config": IdeficsVisionConfig}

    vocab_size: int = 32000
    additional_vocab_size: int = 0
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    dropout: float | int = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    alpha_initializer: str = "zeros"
    alphas_initializer_range: float = 0.0
    alpha_type: str = "float"
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    cross_layer_interval: int = 1
    qk_layer_norms: bool = False
    freeze_text_layers: bool = True
    freeze_text_module_exceptions: list | tuple = ()
    freeze_lm_head: bool = False
    freeze_vision_layers: bool = True
    freeze_vision_module_exceptions: list | tuple = ()
    use_resampler: bool = False
    vision_config: dict | PreTrainedConfig | None = None
    perceiver_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.perceiver_config is None:
            self.perceiver_config = IdeficsPerceiverConfig()
        elif isinstance(self.perceiver_config, dict):
            self.perceiver_config = IdeficsPerceiverConfig(**self.perceiver_config)

        if self.vision_config is None:
            self.vision_config = IdeficsVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = IdeficsVisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["IdeficsConfig", "IdeficsPerceiverConfig", "IdeficsVisionConfig"]
