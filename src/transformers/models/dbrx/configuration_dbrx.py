
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@strict
@auto_docstring(
    custom_intro="This config is used to instantiate attention layers.",
    checkpoint="transformers-community/dbrx-instruct",
)
class DbrxAttentionConfig(PreTrainedConfig):

    base_config_key = "attn_config"

    attn_pdrop: float | int = 0.0
    clip_qkv: int | float | None = None
    kv_n_heads: int = 1


@strict
@auto_docstring(
    custom_intro="This config is used to instantiate feedforward layers.",
    checkpoint="transformers-community/dbrx-instruct",
)
class DbrxFFNConfig(PreTrainedConfig):

    base_config_key = "ffn_config"

    hidden_size: int = 6144
    ffn_act_fn: dict | None = None
    ffn_hidden_size: int = 3584
    moe_num_experts: int = 4
    moe_top_k: int = 1
    moe_jitter_eps: float | None = None
    moe_loss_weight: float = 0.01
    moe_normalize_expert_weights: float | None = 1.0

    def __post_init__(self, **kwargs):
        if self.ffn_act_fn is None:
            self.ffn_act_fn = {"name": "silu"}

        for k in [
            "model_type",
            "attn_implementation",
            "experts_implementation",
            "transformers_version",
            "_commit_hash",
            "torch_dtype",
            "dtype",
        ]:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f"Found unknown {kwargs=}")

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="transformers-community/dbrx-instruct")
@strict
class DbrxConfig(PreTrainedConfig):

    model_type = "dbrx"
    sub_configs = {"attn_config": DbrxAttentionConfig, "ffn_config": DbrxFFNConfig}
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
        "max_position_embeddings": "max_seq_len",
    }

    d_model: int | None = 2048
    n_heads: int | None = 16
    n_layers: int | None = 24
    max_seq_len: int | None = 2048
    vocab_size: int = 32000
    resid_pdrop: float | None = 0.0
    emb_pdrop: float | None = 0.0
    attn_config: DbrxAttentionConfig | dict | None = None
    ffn_config: DbrxFFNConfig | dict | None = None
    use_cache: bool = True
    initializer_range: float = 0.02
    output_router_logits: bool | None = False
    rope_parameters: RopeParameters | dict | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.attn_config is None:
            self.attn_config = DbrxAttentionConfig()
        elif isinstance(self.attn_config, dict):
            self.attn_config = DbrxAttentionConfig(**self.attn_config)

        if self.ffn_config is None:
            self.ffn_config = DbrxFFNConfig()
        elif isinstance(self.ffn_config, dict):
            self.ffn_config = DbrxFFNConfig(**self.ffn_config)

        self.num_key_value_heads = self.attn_config.kv_n_heads
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["DbrxConfig"]
