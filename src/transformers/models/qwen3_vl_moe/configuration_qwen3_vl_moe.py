from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="Qwen/Qwen3-VL-30B-A3B-Instruct")
@strict
class Qwen3VLMoeTextConfig(PreTrainedConfig):

    model_type = "qwen3_vl_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_experts": "num_local_experts",
    }
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 151936
    hidden_size: int = 2048

    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 1408
    num_experts_per_tok: int = 4
    num_experts: int = 60
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list[int] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    base_config_key = "text_config"
    default_theta = 500000.0
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}
    head_dim: int | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        self.sliding_window = None
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen3-VL-30B-A3B-Instruct")
@strict
class Qwen3VLMoeVisionConfig(PreTrainedConfig):

    model_type = "qwen3_vl_moe_vision"
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


@auto_docstring(checkpoint="Qwen/Qwen3-VL-30B-A3B-Instruct")
@strict
class Qwen3VLMoeConfig(PreTrainedConfig):

    model_type = "qwen3_vl_moe"
    sub_configs = {"vision_config": Qwen3VLMoeVisionConfig, "text_config": Qwen3VLMoeTextConfig}
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
            if self.vision_config.get("model_type") == "qwen3_vl_moe":
                self.vision_config["model_type"] = "qwen3_vl_moe_vision"
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        super().__post_init__(**kwargs)


__all__ = ["Qwen3VLMoeConfig", "Qwen3VLMoeTextConfig", "Qwen3VLMoeVisionConfig"]
