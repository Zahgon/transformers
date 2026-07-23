
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/recurrentgemma-2b")
@strict
class RecurrentGemmaConfig(PreTrainedConfig):

    model_type = "recurrent_gemma"
    attribute_map = {"sliding_window": "attention_window_size"}

    num_hidden_layers: int = 26
    vocab_size: int = 256000
    hidden_size: int = 2560
    intermediate_size: int = 3 * 2560
    num_attention_heads: int = 10
    lru_width: int | None = None
    attention_window_size: int = 2048
    conv1d_width: int = 4
    logits_soft_cap: float = 30.0
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    hidden_activation: str = "gelu_pytorch_tanh"
    rope_parameters: RopeParameters | dict | None = None
    block_types: list[str] | tuple[str, ...] | None = ("recurrent", "recurrent", "attention")
    attention_dropout: float | int = 0.0
    num_key_value_heads: int | None = None
    attention_bias: bool = False
    w_init_variance_scale: float = 0.01
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.lru_width = self.lru_width if self.lru_width is not None else self.hidden_size
        self.block_types = list(self.block_types)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = (
            self.num_key_value_heads if self.num_key_value_heads is not None else self.num_attention_heads
        )
        self.final_w_init_variance_scale = 2.0 / self.num_hidden_layers
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def layers_block_type(self):
        pass


__all__ = ["RecurrentGemmaConfig"]
