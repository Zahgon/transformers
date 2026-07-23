

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="LiquidAI/LFM2-1.2B")
@strict
class Lfm2Config(PreTrainedConfig):

    model_type = "lfm2"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    vocab_size: int = 65536
    hidden_size: int = 2560
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 128_000
    initializer_range: float = 0.02
    norm_eps: float = 0.00001
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    conv_bias: bool = False
    conv_L_cache: int = 3
    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float | int = 1.0
    block_auto_adjust_ff_dim: bool = True
    full_attn_idxs: list[int] | None = None
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.full_attn_idxs = (
                self.full_attn_idxs if self.full_attn_idxs is not None else list(range(self.num_hidden_layers))
            )
            self.layer_types = [
                "full_attention" if i in self.full_attn_idxs else "conv" for i in range(self.num_hidden_layers)
            ]

        self.tie_word_embeddings = kwargs.pop("tie_embedding", self.tie_word_embeddings)
        self.intermediate_size = kwargs.pop("block_ff_dim", self.intermediate_size)
        super().__post_init__(**kwargs)


__all__ = ["Lfm2Config"]
