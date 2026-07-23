from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..siglip import SiglipVisionConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/gemma-3-4b-it")
@strict
class Gemma3TextConfig(PreTrainedConfig):

    model_type = "gemma3_text"
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
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 262_208
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
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
    query_pre_attn_scalar: int = 256
    sliding_window: int | None = 4096
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = None
    attn_logit_softcapping: float | None = None
    use_bidirectional_attention: bool | None = False
    default_theta = {"global": 1_000_000.0, "local": 10_000.0}

    def __post_init__(self, **kwargs):
        if self.use_bidirectional_attention:
            self.sliding_window = (self.sliding_window // 2) + 1  # due to fa we set exclusive bounds

        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)

        if self.rope_parameters.get("full_attention") is None:
            self.rope_parameters["full_attention"] = {"rope_type": "default"}
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_theta", self.default_theta["global"])
        )
        if self.rope_parameters.get("sliding_attention") is None:
            self.rope_parameters["sliding_attention"] = {"rope_type": "default"}
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_local_base_freq", self.default_theta["local"])
        )

        self.standardize_rope_params()
        return kwargs


@auto_docstring(checkpoint="google/gemma-3-4b-it")
@strict
class Gemma3Config(PreTrainedConfig):

    model_type = "gemma3"
    attribute_map = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }
    sub_configs = {
        "text_config": Gemma3TextConfig,
        "vision_config": SiglipVisionConfig,
    }

    text_config: Gemma3TextConfig | dict[str, Any] | None = None
    vision_config: SiglipVisionConfig | dict[str, Any] | None = None
    mm_tokens_per_image: int | None = 256
    boi_token_index: int | None = 255_999
    eoi_token_index: int | None = 256_000
    image_token_index: int | None = 262_144
    initializer_range: float | None = 0.02
    tie_word_embeddings: bool | None = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Gemma3TextConfig()
            logger.info("text_config is None, using default Gemma3TextConfig text config.")
        elif isinstance(self.text_config, dict):
            self.text_config = Gemma3TextConfig(**self.text_config)

        if isinstance(self.vision_config, dict):
            self.vision_config = SiglipVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = SiglipVisionConfig()
            logger.info("vision_config is None, using default SiglipVisionConfig vision config.")

        super().__post_init__(**kwargs)


__all__ = ["Gemma3Config", "Gemma3TextConfig"]
