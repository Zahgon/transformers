
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="abeja/gpt-neox-japanese-2.7b")
@strict
class GPTNeoXJapaneseConfig(PreTrainedConfig):

    model_type = "gpt_neox_japanese"

    vocab_size: int = 32000
    hidden_size: int = 2560
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_multiple_size: int = 4
    hidden_act: str = "gelu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    bos_token_id: int | None = 31996
    eos_token_id: int | list[int] | None = 31999
    rope_parameters: RopeParameters | dict | None = None
    attention_dropout: float | int = 0.1
    hidden_dropout: float | int = 0.0
    is_decoder: bool = False
    pad_token_id: int | None = None
    tie_word_embeddings: bool = True

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rotary_emb_base", self.default_theta))
        self.rope_parameters.setdefault("partial_rotary_factor", kwargs.pop("rotary_pct", 1.0))
        self.standardize_rope_params()
        return kwargs


__all__ = ["GPTNeoXJapaneseConfig"]
