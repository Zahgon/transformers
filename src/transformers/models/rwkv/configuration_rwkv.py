
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="RWKV/rwkv-4-169m-pile")
@strict
class RwkvConfig(PreTrainedConfig):

    model_type = "rwkv"
    attribute_map = {"max_position_embeddings": "context_length"}

    vocab_size: int = 50277
    context_length: int = 1024
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    attention_hidden_size: int | None = None
    intermediate_size: int | None = None
    layer_norm_epsilon: float = 1e-5
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 0
    rescale_every: int = 6
    tie_word_embeddings: bool = False
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        self.attention_hidden_size = (
            self.attention_hidden_size if self.attention_hidden_size is not None else self.hidden_size
        )
        self.intermediate_size = self.intermediate_size if self.intermediate_size is not None else 4 * self.hidden_size

        super().__post_init__(**kwargs)


__all__ = ["RwkvConfig"]
