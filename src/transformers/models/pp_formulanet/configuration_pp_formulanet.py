
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PPFormulaNet_plus-L_safetensors")
@strict
class PPFormulaNetVisionConfig(PreTrainedConfig):

    base_config_key = "vision_config"
    hidden_size: int = 768
    output_channels: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 512
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-06
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10
    qkv_bias: bool = True
    use_abs_pos: bool = True
    use_rel_pos: bool = True
    window_size: int = 14
    global_attn_indexes: list[int] | tuple[int, ...] = (2, 5, 8, 11)
    mlp_dim: int = 3072

    post_conv_in_channels: int = 256
    post_conv_out_channels: int = 1024
    post_conv_mid_channels: int = 512
    decoder_hidden_size: int = 512


@auto_docstring(checkpoint="PaddlePaddle/PPFormulaNet_plus-L_safetensors")
@strict
class PPFormulaNetTextConfig(PreTrainedConfig):

    model_type = "pp_formulanet"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }
    vocab_size: int = 50000
    max_position_embeddings: int = 2560
    encoder_layers: int = 12
    encoder_attention_heads: int = 16
    decoder_layers: int = 8
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 16
    decoder_layerdrop: float | int = 0.0
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    d_model: int = 512
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    scale_embedding: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    decoder_start_token_id: int | None = 2
    forced_eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    base_config_key = "text_config"


@auto_docstring(checkpoint="PaddlePaddle/PPFormulaNet_plus-L_safetensors")
@strict
class PPFormulaNetConfig(PreTrainedConfig):
    model_type = "pp_formulanet"
    sub_configs = {"text_config": PPFormulaNetTextConfig, "vision_config": PPFormulaNetVisionConfig}

    text_config: dict | PPFormulaNetTextConfig | None = None
    vision_config: dict | PPFormulaNetVisionConfig | None = None
    is_encoder_decoder: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = PPFormulaNetTextConfig(**self.text_config)
        elif self.text_config is None:
            logger.info("text_config is None. Initializing the PPFormulaNetTextConfig with default values.")
            self.text_config = PPFormulaNetTextConfig()

        if isinstance(self.vision_config, dict):
            self.vision_config = PPFormulaNetVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            logger.info("vision_config is None. Initializing the PPFormulaNetVisionConfig with default values.")
            self.vision_config = PPFormulaNetVisionConfig()

        super().__post_init__(**kwargs)


__all__ = ["PPFormulaNetConfig", "PPFormulaNetTextConfig", "PPFormulaNetVisionConfig"]
