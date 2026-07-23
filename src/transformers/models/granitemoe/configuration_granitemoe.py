
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="ibm-granite/granite-speech-3.2-8b")
@strict
class GraniteMoeConfig(PreTrainedConfig):

    model_type = "granitemoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    embedding_multiplier: float | int | None = 1.0
    logits_scaling: float | int | None = 1.0
    residual_multiplier: float | int | None = 1.0
    attention_multiplier: float | int | None = 1.0
    num_local_experts: int | None = 8
    num_experts_per_tok: int | None = 2
    output_router_logits: bool | None = False
    router_aux_loss_coef: float | None = 0.001

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)


__all__ = ["GraniteMoeConfig"]
