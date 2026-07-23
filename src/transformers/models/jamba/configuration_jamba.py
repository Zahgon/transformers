
import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="ai21labs/Jamba-v0.1")
@strict
class JambaConfig(PreTrainedConfig):

    model_type = "jamba"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "num_experts",
    }

    vocab_size: int = 65536
    tie_word_embeddings: bool = False
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    max_position_embeddings: int = 262144
    attention_dropout: float | int = 0.0
    num_experts_per_tok: int = 2
    num_experts: int = 16
    expert_layer_period: int = 2
    expert_layer_offset: int = 1
    attn_layer_period: int = 8
    attn_layer_offset: int = 4
    use_mamba_kernels: bool = True
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: int | str = "auto"
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if self.mamba_dt_rank == "auto" else self.mamba_dt_rank
        super().__post_init__(**kwargs)

    @property
    def layers_block_type(self):
        pass

    @property
    def layer_types(self):
        pass

    @property
    def layers_num_experts(self):
        pass

    def validate_architecture(self):
        pass


__all__ = ["JambaConfig"]
