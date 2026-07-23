import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="tiiuae/falcon-mamba-7b")
@strict
class FalconMambaConfig(PreTrainedConfig):

    model_type = "falcon_mamba"

    vocab_size: int = 50280
    hidden_size: int = 768
    state_size: int = 16
    num_hidden_layers: int = 32
    layer_norm_epsilon: float = 1e-5
    pad_token_id: int | None = 0
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 0
    expand: int = 2
    conv_kernel: int = 4
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.1
    residual_in_fp32: bool = True
    time_step_rank: str | int = "auto"
    time_step_scale: float = 1.0
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_init_scheme: str = "random"
    time_step_floor: float = 1e-4
    rescale_prenorm_residual: bool = False
    use_cache: bool = True

    use_falcon_mambapy: bool = False
    use_associative_scan: bool = True
    tie_word_embeddings: bool = True
    mixer_rms_eps: float = 1e-6

    def __post_init__(self, **kwargs):
        self.intermediate_size = int(self.expand * self.hidden_size)
        self.time_step_rank = (
            math.ceil(self.hidden_size / 16) if self.time_step_rank == "auto" else self.time_step_rank
        )
        super().__post_init__(**kwargs)

    @property
    def layer_types(self):
        pass


__all__ = ["FalconMambaConfig"]
