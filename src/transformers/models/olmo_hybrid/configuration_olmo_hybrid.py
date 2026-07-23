
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, remap_legacy_layer_types
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ...utils.type_validators import interval


@auto_docstring(checkpoint="allenai/Olmo-Hybrid-7B")
@strict
class OlmoHybridConfig(PreTrainedConfig):

    model_type = "olmo_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_split_input",  # input is replicated due to the added norm on q and k
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 100352
    hidden_size: int = 3840
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 30
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 65536
    initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)
    rms_norm_eps: float = 1e-06
    use_cache: bool = True
    pad_token_id: int | None = 100277
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 100257
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    layer_types: list[str] | None = None
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_a_log_min: float = 0.0
    linear_a_log_max: float = 16.0
    linear_dt_min: float = 0.001
    linear_dt_max: float = 0.1
    linear_dt_init_floor: float = 1e-4
    linear_conv_kernel_dim: int = 4
    linear_allow_neg_eigval: bool = True

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["linear_attention"] * int(self.num_hidden_layers)
            for i in range(int(self.num_hidden_layers)):
                if i % 4 == 3:
                    self.layer_types[i] = "full_attention"
            if "full_attention" not in self.layer_types:
                self.layer_types[-1] = "full_attention"
        else:
            self.layer_types = remap_legacy_layer_types(self.layer_types)

        if self.linear_num_key_heads is None:
            self.linear_num_key_heads = self.num_attention_heads
        if self.linear_num_value_heads is None:
            self.linear_num_value_heads = self.num_attention_heads
        if self.linear_key_head_dim is None:
            self.linear_key_head_dim = int(0.75 * self.hidden_size / self.linear_num_key_heads)
        if self.linear_value_head_dim is None:
            self.linear_value_head_dim = 2 * self.linear_key_head_dim
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["OlmoHybridConfig"]
