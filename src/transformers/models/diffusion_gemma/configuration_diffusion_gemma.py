
from typing import Any, Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/diffusiongemma-26B-A4B-it")
@strict
class DiffusionGemmaTextConfig(PreTrainedConfig):

    model_type = "diffusion_gemma_text"
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
        "layers.*.experts.gate_up_proj": "packed_colwise",
        "layers.*.experts.down_proj": "rowwise",
        "layers.*.experts": "moe_tp_experts",
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.router": "ep_router",
        "layers.*.experts.gate_up_proj": "grouped_gemm",
        "layers.*.experts.down_proj": "grouped_gemm",
        "layers.*.experts": "moe_tp_experts",
    }

    base_model_pp_plan = {
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
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    sliding_window: int = 512
    layer_types: list[str] | None = None
    final_logit_softcapping = 30.0
    use_bidirectional_attention: Literal["all", "vision"] | None = None
    num_global_key_value_heads: int | None = None
    global_head_dim: int = 512
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None

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


@auto_docstring(checkpoint="google/diffusiongemma-26B-A4B-it")
@strict
class DiffusionGemmaConfig(PreTrainedConfig):

    model_type = "diffusion_gemma"
    sub_configs = {
        "text_config": DiffusionGemmaTextConfig,
        "vision_config": AutoConfig,
    }

    text_config: DiffusionGemmaTextConfig | dict[str, Any] | None = None
    vision_config: PreTrainedConfig | dict[str, Any] | None = None
    boi_token_id: int | None = 255_999
    eoi_token_id: int | None = 258_882
    image_token_id: int | None = 258_880
    initializer_range: float | None = 0.02
    tie_word_embeddings: bool = True
    canvas_length: int | None = 256

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = DiffusionGemmaTextConfig()
            logger.info("text_config is None. Using default DiffusionGemmaTextConfig.")
        elif isinstance(self.text_config, dict):
            self.text_config = DiffusionGemmaTextConfig(**self.text_config)

        if self.vision_config is None:
            logger.info("vision_config is None. DiffusionGemmaEncoderModel.vision_tower will not be initialized.")
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "gemma4_vision")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["DiffusionGemmaTextConfig", "DiffusionGemmaConfig"]
