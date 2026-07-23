
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="funnel-transformer/small")
@strict
class FunnelConfig(PreTrainedConfig):

    model_type = "funnel"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
    }

    vocab_size: int = 30522
    block_sizes: list[int] | tuple[int, ...] = (4, 4, 4)
    block_repeats: list[int] | None = None
    num_decoder_layers: int = 2
    d_model: int = 768
    n_head: int = 12
    d_head: int = 64
    d_inner: int = 3072
    hidden_act: str = "gelu_new"
    hidden_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.0
    initializer_range: float = 0.1
    initializer_std: float | None = None
    layer_norm_eps: float = 1e-9
    pooling_type: str = "mean"
    attention_type: str = "relative_shift"
    separate_cls: bool = True
    truncate_seq: bool = True
    pool_q_only: bool = True
    pad_token_id: int | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.block_repeats = [1] * len(self.block_sizes) if self.block_repeats is None else self.block_repeats
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def num_hidden_layers(self):
        pass

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        raise NotImplementedError(
            "This model does not support the setting of `num_hidden_layers`. Please set `block_sizes`."
        )

    @property
    def num_blocks(self):
        pass

    @num_blocks.setter
    def num_blocks(self, value):
        raise NotImplementedError("This model does not support the setting of `num_blocks`. Please set `block_sizes`.")


__all__ = ["FunnelConfig"]
