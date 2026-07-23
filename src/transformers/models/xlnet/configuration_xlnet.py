
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="xlnet/xlnet-large-cased")
@strict
class XLNetConfig(PreTrainedConfig):

    model_type = "xlnet"
    keys_to_ignore_at_inference = ["mems"]
    attribute_map = {
        "n_token": "vocab_size",  # Backward compatibility
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 32000
    d_model: int = 1024
    n_layer: int = 24
    n_head: int = 16
    d_inner: int = 4096
    d_head: int | None = None
    ff_activation: str = "gelu"
    attn_type: str = "bi"
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    dropout: float | int = 0.1
    mem_len: int | None = 512
    reuse_len: int | None = None
    use_mems_eval: bool = True
    use_mems_train: bool = False
    bi_data: bool = False
    clamp_len: int = -1
    same_length: bool = False
    summary_type: str = "last"
    summary_use_proj: bool = True
    summary_activation: str = "tanh"
    summary_last_dropout: float | int = 0.1
    start_n_top: int = 5
    end_n_top: int = 5
    pad_token_id: int | None = 5
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.d_head = self.d_head or self.d_model // self.n_head
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def max_position_embeddings(self):
        pass

    @max_position_embeddings.setter
    def max_position_embeddings(self, value):
        raise NotImplementedError(
            f"The model {self.model_type} is one of the few models that has no sequence length limit."
        )


__all__ = ["XLNetConfig"]
