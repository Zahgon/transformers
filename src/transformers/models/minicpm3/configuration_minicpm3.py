
import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="openbmb/MiniCPM3-4B")
@strict
class MiniCPM3Config(PreTrainedConfig):

    model_type = "minicpm3"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
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

    vocab_size: int = 73448
    hidden_size: int = 2560
    intermediate_size: int = 6400
    num_hidden_layers: int = 62
    num_attention_heads: int = 40
    num_key_value_heads: int | None = 40
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.1
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pretraining_tp: int | None = 1
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    mlp_bias: bool = False
    head_dim: int | None = None
    kv_lora_rank: int = 256
    q_lora_rank: int | None = 768
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int | None = None
    scale_emb: int | float = 12
    scale_depth: int | float | None = 1.4
    dim_model_base: int | None = 256

    def __post_init__(self, **kwargs):
        self.head_dim = self.qk_rope_head_dim
        if self.v_head_dim is None:
            self.v_head_dim = self.hidden_size // self.num_attention_heads
        if self.scale_depth is None:
            self.scale_depth = math.sqrt(self.num_hidden_layers)
        if self.dim_model_base is None:
            self.dim_model_base = self.hidden_size
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def logits_scaling(self) -> float:
        pass


__all__ = ["MiniCPM3Config"]
