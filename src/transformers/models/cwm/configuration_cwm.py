

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/cwm")
@strict
class CwmConfig(PreTrainedConfig):

    model_type = "cwm"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
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

    vocab_size: int = 128256
    hidden_size: int = 6144
    intermediate_size: int = 21504
    num_hidden_layers: int = 64
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int = 128000
    eos_token_id: int | list[int] | None = None
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False
    head_dim: int = 128
    default_theta = 1_000_000.0
    sliding_window: int = 8192
    layer_types: list[str] | None = None  # ["full_attention"|"sliding_attention"] per layer

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_theta": 1_000_000.0,
                "factor": 16.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            }

        if self.layer_types is None:
            window_pattern = 4
            self.layer_types = [
                ("full_attention" if (i % window_pattern == 0) else "sliding_attention")
                for i in range(self.num_hidden_layers)
            ]

        self.sliding_window = int(self.sliding_window) if self.sliding_window else None
        self.layer_types = list(self.layer_types)
        self.eos_token_id = self.eos_token_id if self.eos_token_id is not None else [128001, 128008, 128009]
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["CwmConfig"]
