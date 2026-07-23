from typing import Any, Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import (
    auto_docstring,
    logging,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/gemma-4-12B-it")
@strict
class Gemma4UnifiedAudioConfig(PreTrainedConfig):

    model_type = "gemma4_unified_audio"

    audio_embed_dim: int = 640
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    @property
    def hidden_size(self):
        pass

    @property
    def output_proj_dims(self):
        pass

    @property
    def audio_samples_per_token(self):
        pass

    @hidden_size.setter
    def hidden_size(self, value):
        pass

    @output_proj_dims.setter
    def output_proj_dims(self, value):
        pass

    @audio_samples_per_token.setter
    def audio_samples_per_token(self, value):
        pass


@auto_docstring(checkpoint="google/gemma-4-12B-it")
@strict
class Gemma4UnifiedTextConfig(PreTrainedConfig):

    model_type = "gemma4_unified_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_ep_plan = None  # No MoE
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 262_144
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 262_144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0

    sliding_window: int = 1024
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = None
    use_bidirectional_attention: Literal["all", "vision"] | None = "vision"
    num_global_key_value_heads: int | None = None
    global_head_dim: int = 512
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False

    def __post_init__(self, **kwargs):
        if self.use_bidirectional_attention == "all":
            self.is_causal = False
            self.sliding_window = (self.sliding_window // 2) + 1  # due to fa we set exclusive bounds

        if self.layer_types is None:
            sliding_window_pattern = 6  # by default 5:1
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.layer_types and (last_layer_type := self.layer_types[-1]) != "full_attention":
            logger.warning(
                f"Last layer must use `full_attention`, but got `{last_layer_type}`. Forcing last layer to `full_attention`."
            )
            self.layer_types[-1] = "full_attention"

        default_rope_params: dict[Literal["full_attention", "sliding_attention"] : dict[str, Any]] = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0},
        }
        if self.rope_parameters is None:
            self.rope_parameters = default_rope_params

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        return kwargs


@auto_docstring(checkpoint="google/gemma-4-12B-it")
@strict
class Gemma4UnifiedVisionConfig(PreTrainedConfig):

    model_type = "gemma4_unified_vision"

    patch_size: int = 16
    pooling_kernel_size: int = 3
    mm_embed_dim: int = 3840
    mm_posemb_size: int = 1120
    rms_norm_eps: float = 1e-6
    output_proj_dims: int = 3840
    initializer_range: float = 0.02

    @property
    def model_patch_size(self):
        pass

    @model_patch_size.setter
    def model_patch_size(self, value):
        pass


@auto_docstring(checkpoint="google/gemma-4-12B-it")
@strict
class Gemma4UnifiedConfig(PreTrainedConfig):

    model_type = "gemma4_unified"
    sub_configs = {
        "text_config": Gemma4UnifiedTextConfig,
        "vision_config": Gemma4UnifiedVisionConfig,
        "audio_config": Gemma4UnifiedAudioConfig,
    }

    text_config: Gemma4UnifiedTextConfig | dict[str, Any] | None = None
    vision_config: Gemma4UnifiedVisionConfig | dict[str, Any] | None = None
    audio_config: Gemma4UnifiedAudioConfig | dict[str, Any] | None = None
    boi_token_id: int | None = 255_999
    eoi_token_id: int | None = 258_882
    image_token_id: int | None = 258_880
    video_token_id: int | None = 258_884
    boa_token_id: int | None = 256_000
    eoa_token_index: int | None = 258_883
    audio_token_id: int | None = 258_881
    initializer_range: float | None = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Gemma4UnifiedTextConfig()
            logger.info("text_config is None. Using default Gemma4UnifiedTextConfig.")
        elif isinstance(self.text_config, dict):
            self.text_config = Gemma4UnifiedTextConfig(**self.text_config)

        if self.vision_config is None:
            logger.info("vision_config is None. Gemma4UnifiedModel.vision_tower will not be initialized.")
        if isinstance(self.vision_config, dict):
            self.vision_config = Gemma4UnifiedVisionConfig(**self.vision_config)

        if self.audio_config is None:
            logger.info("audio_config is None. Gemma4UnifiedModel.audio_tower will not be initialized.")
        if isinstance(self.audio_config, dict):
            self.audio_config = Gemma4UnifiedAudioConfig(**self.audio_config)

        super().__post_init__(**kwargs)


__all__ = ["Gemma4UnifiedAudioConfig", "Gemma4UnifiedConfig", "Gemma4UnifiedTextConfig", "Gemma4UnifiedVisionConfig"]
