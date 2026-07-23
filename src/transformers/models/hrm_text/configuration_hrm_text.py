
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ...utils.generic import is_flash_attention_requested, split_attention_implementation
from ...utils.type_validators import interval


@auto_docstring(checkpoint="sapientinc/HRM-Text-1B")
@strict
class HrmTextConfig(PreTrainedConfig):

    model_type = "hrm_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        **{f"{stack}.layers.*.self_attn.q_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.self_attn.k_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.self_attn.v_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.self_attn.gate_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.self_attn.o_proj": "rowwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.mlp.gate_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.mlp.up_proj": "colwise" for stack in ("L_module", "H_module")},
        **{f"{stack}.layers.*.mlp.down_proj": "rowwise" for stack in ("L_module", "H_module")},
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 151808
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    mlp_bias: bool = False
    head_dim: int = 128

    H_cycles: int = 2
    L_cycles: int = 3
    L_bp_cycles: list[int] | None = None
    embedding_scale: float | None = None
    prefix_lm: bool = True
    num_layers_per_stack: int | None = None  # Usually inferred in post init

    def __post_init__(self, **kwargs):
        if self.L_bp_cycles is None:
            self.L_bp_cycles = [2]

        if self.embedding_scale is None:
            self.embedding_scale = 1.0 / self.initializer_range

        if self.num_layers_per_stack is None:
            self.num_layers_per_stack = self.num_hidden_layers
            self.num_hidden_layers = self.num_layers_per_stack * self.H_cycles * (self.L_cycles + 1)

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def _attn_implementation(self):
        pass

    @_attn_implementation.setter
    def _attn_implementation(self, value: str | dict | None):
        pass


__all__ = ["HrmTextConfig"]
