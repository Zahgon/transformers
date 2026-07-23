
import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="state-spaces/mamba2-2.8b")
@strict
class Mamba2Config(PreTrainedConfig):

    model_type = "mamba2"

    num_heads: int = 128
    head_dim: int = 64
    vocab_size: int = 32768
    hidden_size: int = 4096
    state_size: int = 128
    num_hidden_layers: int = 64
    layer_norm_epsilon: float = 1e-5
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    expand: int = 2
    conv_kernel: int = 4
    n_groups: int = 8
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.1
    residual_in_fp32: bool = True
    time_step_rank: str | int = "auto"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: list[float] | tuple[float, ...] = (0.0, float("inf"))
    rescale_prenorm_residual: bool = False
    use_cache: bool = True
    rms_norm: bool = True
    chunk_size: int = 256
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        self.time_step_rank = (
            math.ceil(self.hidden_size / 16) if self.time_step_rank == "auto" else self.time_step_rank
        )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def layer_types(self):
        pass


__all__ = ["Mamba2Config"]
