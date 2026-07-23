
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="EleutherAI/gpt-neo-1.3B")
@strict
class GPTNeoConfig(PreTrainedConfig):

    model_type = "gpt_neo"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    hidden_size: int = 2048
    num_layers: int = 24
    attention_types: list | tuple | None = None
    num_heads: int = 16
    intermediate_size: int | None = None
    window_size: int = 256
    activation_function: str = "gelu_new"
    resid_dropout: float | int = 0.0
    embed_dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    classifier_dropout: float | int = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    pad_token_id: int | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.attention_types is None:
            self.attention_types = [[["global", "local"], 12]]
        self.attention_layers = self.expand_attention_types_params(self.attention_types)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @staticmethod
    def expand_attention_types_params(attention_types):
        attentions = []
        for item in attention_types:
            for _ in range(item[1]):
                attentions.extend(item[0])
        return attentions


def custom_unfold(input, dimension, size, step):
    pass


def custom_get_block_length_and_num_blocks(seq_length, window_size):
    pass


__all__ = ["GPTNeoConfig"]
