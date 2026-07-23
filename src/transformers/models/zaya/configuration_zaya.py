
from typing import Any, Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="Zyphra/ZAYA1-8B")
@strict
class ZayaConfig(PreTrainedConfig):

    model_type = "zaya"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    vocab_size: int = 262272
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    sliding_window: int | None = None
    attention_dropout: float | int = 0.0
    moe_intermediate_size: int = 2048

    num_experts_per_tok: int = 1
    num_experts: int = 16
    output_router_logits: bool = False
    layer_types: list[str] | None = None
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 106

    head_dim: int = 128
    attention_bias: bool = False

    lm_head_bias: bool = False
    router_hidden_size: int = 256
    cca_time0: int = 2
    cca_time1: int = 2

    def __post_init__(self, **kwargs):
        self.layer_types = ["hybrid"] * self.num_hidden_layers if self.layer_types is None else list(self.layer_types)

        default_rope_params: dict[Literal["hybrid", "hybrid_sliding"], dict[str, Any]] = {
            "hybrid": {
                "rope_type": "default",
                "rope_theta": 5_000_000.0,
                "partial_rotary_factor": 0.5,
            },
            "hybrid_sliding": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 0.5,
            },
        }
        if self.rope_parameters is None:
            self.rope_parameters = default_rope_params

        super().__post_init__(**kwargs, ignore_keys_at_rope_validation={"hybrid", "hybrid_sliding"})

    def convert_rope_params_to_dict(self, **kwargs):
        return kwargs

    def validate_architecture(self):
        pass


__all__ = ["ZayaConfig"]
