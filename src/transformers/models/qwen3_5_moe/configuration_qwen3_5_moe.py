from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, remap_legacy_layer_types
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="Qwen/Qwen3.5-35B-A3B")
@strict
class Qwen3_5MoeTextConfig(PreTrainedConfig):

    model_type = "qwen3_5_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
        "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_z": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_b": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_a": "colwise_gather_output",
        "layers.*.linear_attn.out_proj": "colwise_gather_output",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 8
    num_experts: int = 256
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    base_config_key = "text_config"
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.25)  # assign default for BC
        if self.layer_types is None:
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = remap_legacy_layer_types(self.layer_types)

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen3.5-35B-A3B")
@strict
class Qwen3_5MoeVisionConfig(PreTrainedConfig):

    model_type = "qwen3_5_moe_vision"
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
    initializer_range: float = 0.02


@auto_docstring(checkpoint="Qwen/Qwen3.5-35B-A3B")
@strict
class Qwen3_5MoeConfig(PreTrainedConfig):

    model_type = "qwen3_5_moe"
    sub_configs = {"vision_config": Qwen3_5MoeVisionConfig, "text_config": Qwen3_5MoeTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            if self.vision_config.get("model_type") == "qwen3_5_moe":
                self.vision_config["model_type"] = "qwen3_5_moe_vision"
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        super().__post_init__(**kwargs)


__all__ = ["Qwen3_5MoeConfig", "Qwen3_5MoeTextConfig", "Qwen3_5MoeVisionConfig"]
