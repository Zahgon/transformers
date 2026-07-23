

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ...utils.type_validators import interval


@auto_docstring(checkpoint="bosonai/higgs-audio-v2-generation-3B-base")
@strict
class HiggsAudioV2Config(PreTrainedConfig):

    model_type = "higgs_audio_v2"
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
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 128001
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 128009
    pretraining_tp: int | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    mlp_bias: bool = False
    head_dim: int | None = 128
    num_codebooks: int = 8
    codebook_size: int = 1024
    audio_token_id: int = 128016
    audio_bos_token_id: int = 128013
    audio_delay_token_id: int = 128014
    audio_stream_bos_id: int = 1024
    audio_stream_eos_id: int = 1025

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "factor": 32.0,
                "rope_theta": 500000.0,
                "high_freq_factor": 0.5,
                "low_freq_factor": 0.125,
                "original_max_position_embeddings": 1024,
                "rope_type": "llama3",
            }
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["HiggsAudioV2Config"]
