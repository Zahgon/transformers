
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="westlake-repl/Evolla-10B-hf")
@strict
class SaProtConfig(PreTrainedConfig):

    vocab_size: int = 446
    mask_token_id: int = 4
    pad_token_id: int = 1
    hidden_size: int = 1280
    num_hidden_layers: int = 33
    num_attention_heads: int = 20
    intermediate_size: int = 5120
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 1026
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-05
    position_embedding_type: str = "rotary"
    rope_theta: float = 10000.0
    emb_layer_norm_before: bool = False
    token_dropout: bool = True
    is_decoder: bool = False
    add_cross_attention: bool = False


@auto_docstring(checkpoint="westlake-repl/Evolla-10B-hf")
@strict
class EvollaConfig(PreTrainedConfig):

    model_type = "evolla"
    sub_configs = {"protein_encoder_config": SaProtConfig}
    default_theta = 500000.0

    protein_encoder_config: dict | PreTrainedConfig | None = None
    vocab_size: int = 128256  # llama vocab size
    hidden_size: int = 4096  # llama hidden size
    intermediate_size: int = 14336  # llama intermediate size
    num_hidden_layers: int = 32  # llama num layers
    num_attention_heads: int = 32  # llama num heads
    num_key_value_heads: int | None = 8  # llama num key-value heads
    hidden_act: str = "silu"  # llama activation function
    max_position_embeddings: int = 8192  # llama rope max length
    rms_norm_eps: float = 1e-05
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    mlp_bias: bool = False
    aligner_ffn_mult: int | None = 4
    aligner_enable_bias: bool | None = True
    aligner_attention_probs_dropout_prob: float | None = 0.1
    aligner_num_add_layers: int | None = 8
    resampler_depth: int | None = 6
    resampler_dim_head: int | None = 64
    resampler_heads: int | None = 8
    resampler_num_latents: int | None = 64
    resampler_ff_mult: int | None = 4
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = 128009
    use_cache: bool = False
    tie_word_embeddings: bool = False
    is_decoder: bool | None = False
    add_cross_attention: bool | None = False

    def __post_init__(self, **kwargs):
        if self.protein_encoder_config is None:
            self.protein_encoder_config = SaProtConfig()
            logger.info("`protein_encoder_config` is `None`. Initializing the `SaProtConfig` with default values.")
        elif isinstance(self.protein_encoder_config, dict):
            self.protein_encoder_config = SaProtConfig(**self.protein_encoder_config)
        super().__post_init__(**kwargs)


__all__ = ["EvollaConfig"]
