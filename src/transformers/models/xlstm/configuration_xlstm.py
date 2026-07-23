

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, is_xlstm_available


if is_xlstm_available():
    from xlstm.xlstm_large.model import (
        BackendModeType,
        ChunkwiseKernelType,
        DtypeType,
        SequenceKernelType,
        StepKernelType,
        WeightModeType,
        round_up_to_next_multiple_of,
        xLSTMLargeConfig,
    )

    external_xlstm = True
else:
    from typing import Literal

    BackendModeType = Literal["train", "train_with_padding", "inference"]
    ChunkwiseKernelType = Literal[
        "chunkwise--native_autograd",
        "parallel--native_autograd",
    ]
    DtypeType = Literal["float32", "bfloat16", "float16"]
    SequenceKernelType = Literal["native_sequence__native"]
    StepKernelType = Literal["native"]
    WeightModeType = Literal["single", "fused"]

    def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
        """Rounds up x to the next multiple of multiple_of."""
        return int(((x + multiple_of - 1) // multiple_of) * multiple_of)

    external_xlstm = False


@auto_docstring(checkpoint="NX-AI/xLSTM-7b")
@strict
class xLSTMConfig(PreTrainedConfig):

    model_type = "xlstm"

    vocab_size: int = 50304
    hidden_size: int = 4096
    embedding_dim: int | None = None
    num_hidden_layers: int = 32
    num_blocks: int | None = None
    num_heads: int = 8
    use_bias: bool = False
    norm_reduction_force_float32: bool = True
    tie_word_embeddings: bool = False
    add_out_norm: bool = True
    norm_eps: float = 1e-6
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    chunkwise_kernel: ChunkwiseKernelType = "chunkwise--native_autograd"
    sequence_kernel: SequenceKernelType = "native_sequence__native"
    step_kernel: StepKernelType = "native"
    mode: BackendModeType = "inference"
    chunk_size: int = 64
    return_last_states: bool = True
    autocast_kernel_dtype: DtypeType = "bfloat16"
    eps: float = 1e-6
    inference_state_dtype: DtypeType = "float32"
    ffn_proj_factor: float = 2.667
    ffn_round_up_to_multiple_of: int = 64
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0
    weight_mode: WeightModeType = "single"
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    max_inference_chunksize: int = 16384

    def __post_init__(self, **kwargs):
        self.hidden_size = self.hidden_size if self.hidden_size is not None else self.embedding_dim
        self.embedding_dim = self.embedding_dim if self.embedding_dim is not None else self.hidden_size
        self.num_hidden_layers = self.num_hidden_layers if self.num_hidden_layers is not None else self.num_blocks
        self.num_blocks = self.num_blocks if self.num_blocks is not None else self.num_hidden_layers
        super().__post_init__(**kwargs)

    @property
    def qk_dim(self):
        pass

    @property
    def v_dim(self):
        pass

    @property
    def qk_head_dim(self):
        pass

    @property
    def v_head_dim(self):
        pass

    def to_xlstm_block_config(self):
        pass


__all__ = ["xLSTMConfig"]
