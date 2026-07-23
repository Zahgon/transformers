
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@strict
@auto_docstring(
    custom_intro="""
    The BambaModel is a hybrid [mamba2](https://github.com/state-spaces/mamba) architecture with SwiGLU.
    The checkpoints are  jointly trained by IBM, Princeton, and UIUC.
    """,
    checkpoint="ibm-fms/Bamba-9.8b-2.2T-hf",
)
class BambaConfig(PreTrainedConfig):

    model_type = "bamba"
    attribute_map = {"layer_types": "layers_block_type"}
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 128000
    tie_word_embeddings: bool = False
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    num_logits_to_keep: int | None = 1
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    max_position_embeddings: int = 262144
    attention_dropout: float | int | None = 0.0
    attn_layer_indices: list[int] | None = None
    mamba_n_heads: int | None = 128
    mamba_d_head: str | int | None = "auto"
    mamba_n_groups: int | None = 1
    mamba_d_state: int | None = 256
    mamba_d_conv: int | None = 4
    mamba_expand: int | None = 2
    mamba_chunk_size: int | None = 256
    mamba_conv_bias: bool | None = True
    mamba_proj_bias: bool | None = False
    time_step_min: float | None = 0.001
    time_step_max: float | None = 0.1
    time_step_limit: list[float] | tuple[float, float] | None = (0.0, float("inf"))
    z_loss_coefficient: float | None = 0.0
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.mamba_d_head == "auto":
            self.mamba_d_head = self.mamba_expand * self.hidden_size // self.mamba_n_heads

        self.time_step_limit = tuple(self.time_step_limit) if self.time_step_limit is not None else None
        kwargs["partial_rotary_factor"] = 0.5  # hardcode for BC

        super().__post_init__(**kwargs)

    @property
    def layers_block_type(self):
        pass

    def validate_architecture(self):
        pass


__all__ = ["BambaConfig"]
