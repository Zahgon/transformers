
import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="Zyphra/Zamba-7B-v1")
@strict
class ZambaConfig(PreTrainedConfig):

    model_type = "zamba"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"layer_types": "layers_block_type", "head_dim": "attention_head_dim"}

    vocab_size: int = 32000
    tie_word_embeddings: bool = True
    hidden_size: int = 3712
    attention_hidden_size: int | None = None
    intermediate_size: int = 14848
    num_hidden_layers: int = 76
    num_attention_heads: int = 16
    attention_head_dim: int | None = None
    num_key_value_heads: int = 16
    n_mamba_heads: int = 2
    hidden_act: str = "gelu"
    hidden_mamba_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    num_logits_to_keep: int = 1
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    max_position_embeddings: int = 4096
    attention_dropout: float | int = 0.0
    attn_layer_period: int = 6
    attn_layer_offset: int = 4
    use_mamba_kernels: bool = True
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: str | int = "auto"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    def __post_init__(self, **kwargs):
        self.attention_hidden_size = self.attention_hidden_size or 2 * self.hidden_size
        self.attention_head_dim = self.attention_head_dim or 2 * self.hidden_size // self.num_attention_heads
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if self.mamba_dt_rank == "auto" else self.mamba_dt_rank
        self.layers_block_type = self._layers_block_type(
            self.num_hidden_layers, self.attn_layer_period, self.attn_layer_offset
        )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    def _layers_block_type(self, num_hidden_layers, attn_layer_period, attn_layer_offset):
        layers = [
            "linear_attention",
            "linear_attention",
            "hybrid",
        ] + [
            "hybrid" if i % attn_layer_period == attn_layer_offset else "linear_attention"
            for i in range(num_hidden_layers - 3)
        ]
        return layers


__all__ = ["ZambaConfig"]
