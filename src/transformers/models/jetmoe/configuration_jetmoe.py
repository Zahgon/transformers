
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="jetmoe/jetmoe-8b")
@strict
class JetMoeConfig(PreTrainedConfig):

    model_type = "jetmoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"head_dim": "kv_channels"}

    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 12
    num_key_value_heads: int = 16
    kv_channels: int = 128
    intermediate_size: int = 5632
    max_position_embeddings: int = 4096
    activation_function: str = "silu"
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    output_router_logits: bool = False
    aux_loss_coef: float = 0.01
    use_cache: bool = True
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = None
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.01
    attention_dropout: float | int = 0.0

    def __post_init__(self, **kwargs):
        self.num_attention_heads = self.num_key_value_heads * self.num_experts_per_tok
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["JetMoeConfig"]
