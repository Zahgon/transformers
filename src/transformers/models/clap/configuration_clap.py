
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="laion/clap-htsat-fused")
@strict
class ClapTextConfig(PreTrainedConfig):

    model_type = "clap_text_model"
    base_config_key = "text_config"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    initializer_factor: float = 1.0
    layer_norm_eps: float = 1e-12
    projection_dim: int = 512
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    projection_hidden_act: str = "relu"


@auto_docstring(checkpoint="laion/clap-htsat-fused")
@strict
class ClapAudioConfig(PreTrainedConfig):

    model_type = "clap_audio_model"
    base_config_key = "audio_config"

    window_size: int = 8
    num_mel_bins: int = 64
    spec_size: int = 256
    hidden_act: str = "gelu"
    patch_size: int | list[int] | tuple[int, int] = 4
    patch_stride: int | list[int] | tuple[int, ...] = (4, 4)
    num_classes: int = 527
    hidden_size: int = 768
    projection_dim: int = 512
    depths: list[int] | tuple[int, ...] = (2, 2, 6, 2)
    num_attention_heads: list[int] | tuple[int, ...] = (4, 8, 16, 32)
    enable_fusion: bool = False
    hidden_dropout_prob: float | int = 0.1
    fusion_type: str | None = None
    patch_embed_input_channels: int = 1
    flatten_patch_embeds: bool = True
    patch_embeds_hidden_size: int = 96
    enable_patch_layer_norm: bool = True
    drop_path_rate: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    qkv_bias: bool = True
    mlp_ratio: float = 4.0
    aff_block_r: int = 4
    num_hidden_layers: int = 4
    projection_hidden_act: str = "relu"
    layer_norm_eps: float = 1e-5
    initializer_factor: float = 1.0


@auto_docstring(checkpoint="laion/clap-htsat-fused")
@strict
class ClapConfig(PreTrainedConfig):

    model_type = "clap"
    sub_configs = {"text_config": ClapTextConfig, "audio_config": ClapAudioConfig}

    text_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = None
    logit_scale_init_value: float = 1 / 0.07
    projection_dim: int = 512
    projection_hidden_act: str = "relu"
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = ClapTextConfig()
            logger.info("`text_config` is `None`. initializing the `ClapTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = ClapTextConfig(**self.text_config)

        if self.audio_config is None:
            self.audio_config = ClapAudioConfig()
            logger.info("`audio_config` is `None`. initializing the `ClapAudioConfig` with default values.")
        elif isinstance(self.audio_config, dict):
            self.audio_config = ClapAudioConfig(**self.audio_config)

        self.text_config.projection_dim = self.projection_dim
        self.audio_config.projection_dim = self.projection_dim

        self.text_config.projection_hidden_act = self.projection_hidden_act
        self.audio_config.projection_hidden_act = self.projection_hidden_act
        self.hidden_size = self.text_config.hidden_size
        self.num_hidden_layers = self.text_config.num_hidden_layers + len(self.audio_config.depths)
        super().__post_init__(**kwargs)


__all__ = ["ClapAudioConfig", "ClapConfig", "ClapTextConfig"]
