

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, remap_legacy_layer_types
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="Zyphra/Zamba2-2.7B")
@strict
class Zamba2Config(PreTrainedConfig):

    model_type = "zamba2"
    attribute_map = {"layer_types": "layers_block_type", "head_dim": "attention_head_dim"}
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    hidden_size: int = 2560
    num_hidden_layers: int = 54
    layers_block_type: list[str] | None = None
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_ngroups: int = 1
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: list[float] | tuple[float, ...] | None = None
    n_mamba_heads: int = 8
    use_mamba_kernels: bool = True
    use_conv_bias: bool = True
    chunk_size: int = 256
    use_mem_eff_path: bool = False
    add_bias_linear: bool = False
    intermediate_size: int | None = None
    hidden_act: str = "gelu"
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    attention_dropout: float | int = 0.0
    num_mem_blocks: int = 1
    use_shared_attention_adapter: bool = False
    adapter_rank: int = 128
    use_mem_rope: bool = False
    rope_parameters: RopeParameters | dict | None = None
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    num_logits_to_keep: int = 1
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    use_long_context: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.intermediate_size = self.intermediate_size or 4 * self.hidden_size
        self.attention_hidden_size = 2 * self.hidden_size
        self.attention_head_dim = 2 * self.hidden_size // self.num_attention_heads
        self.mamba_headdim = int(self.mamba_expand * self.hidden_size) // self.n_mamba_heads
        if self.use_long_context:
            self.max_position_embeddings = 16384

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.kv_channels = self.hidden_size // self.num_attention_heads
        self.num_query_groups = self.num_attention_heads

        if self.layers_block_type is None:
            self.layers_block_type = (
                ["linear_attention"]
                + (["linear_attention"] * 5 + ["hybrid"]) * 7
                + ["linear_attention"] * 4
                + ["hybrid"]
                + ["linear_attention"] * 3
                + ["hybrid"]
                + ["linear_attention"] * 2
            )
        else:
            self.layers_block_type = remap_legacy_layer_types(self.layers_block_type)
        self.hybrid_layer_ids = [index for index, type in enumerate(self.layers_block_type) if type == "hybrid"]
        super().__post_init__(**kwargs)


__all__ = ["Zamba2Config"]
