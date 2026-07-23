from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniVisionEncoderConfig(PreTrainedConfig):

    model_type = "qwen2_5_omni_vision_encoder"
    base_config_key = "vision_config"

    depth: int = 32
    hidden_size: int = 3584
    hidden_act: str = "silu"
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    window_size: int = 112
    out_hidden_size: int = 3584
    fullatt_block_indexes: list[int] | tuple[int, ...] = (7, 15, 23, 31)
    initializer_range: float = 0.02


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniAudioEncoderConfig(PreTrainedConfig):

    model_type = "qwen2_5_omni_audio_encoder"
    attribute_map = {
        "num_hidden_layers": "encoder_layers",
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "intermediate_size": "encoder_ffn_dim",
    }

    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    d_model: int = 1280
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    activation_function: str = "gelu"
    activation_dropout: float | int = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500

    n_window: int = 100
    output_dim: int = 3584


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniTextConfig(PreTrainedConfig):

    model_type = "qwen2_5_omni_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    ignore_keys_at_rope_validation = {"mrope_section"}

    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int | None = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_parameters: RopeParameters | dict | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = 32768
    max_window_layers: int = 28
    layer_types: list[str] | None = None
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniThinkerConfig(PreTrainedConfig):

    model_type = "qwen2_5_omni_thinker"
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {
        "audio_config": Qwen2_5OmniAudioEncoderConfig,
        "vision_config": Qwen2_5OmniVisionEncoderConfig,
        "text_config": Qwen2_5OmniTextConfig,
    }

    audio_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_index: int = 151646
    image_token_index: int = 151655
    video_token_index: int = 151656
    position_id_per_seconds: int = 25
    seconds_per_chunk: int = 2
    audio_start_token_id: int = 151647
    audio_end_token_id: int = 151648
    user_token_id: int = 872
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = Qwen2_5OmniVisionEncoderConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Qwen2_5OmniVisionEncoderConfig()

        if isinstance(self.audio_config, dict):
            self.audio_config = Qwen2_5OmniAudioEncoderConfig(**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = Qwen2_5OmniAudioEncoderConfig()

        if isinstance(self.text_config, dict):
            self.text_config = Qwen2_5OmniTextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = Qwen2_5OmniTextConfig()

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniTalkerConfig(PreTrainedConfig):

    model_type = "qwen2_5_omni_talker"
    default_theta = 1000000.0
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
        "audio_token_id": "audio_token_index",
    }
    ignore_keys_at_rope_validation = {"mrope_section"}

    audio_token_index: int = 151646
    image_token_index: int = 151655
    video_token_index: int = 151656
    vocab_size: int = 8448
    tts_text_start_token_id: int = 151860
    tts_text_end_token_id: int = 151861
    tts_text_pad_token_id: int = 151859
    tts_codec_start_token_id: int = 8293
    tts_codec_end_token_id: int = 8294
    tts_codec_pad_token_id: int = 8292
    tts_codec_mask_token_id: int = 8296
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    embedding_size: int = 3584
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-06
    head_dim: int = 128
    use_cache: bool = True
    tie_word_embeddings: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = 32768
    max_window_layers: int = 28
    attention_dropout: float | int = 0.0
    rope_parameters: RopeParameters | dict | None = None
    position_id_per_seconds: int = 25
    seconds_per_chunk: int = 2
    audio_start_token_id: int = 151647
    audio_end_token_id: int = 151648
    initializer_range: float = 0.02
    spatial_merge_size: int = 2
    layer_types: list[str] | None = None
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniDiTConfig(PreTrainedConfig):

    model_type = "qwen2_5_omni_dit"

    hidden_size: int = 1024
    num_hidden_layers: int = 22
    num_attention_heads: int = 16
    ff_mult: int = 2
    emb_dim: int = 512
    head_dim: int = 64
    rope_parameters: RopeParameters | dict | None = None
    max_position_embeddings: int = 32768
    block_size: int = 24
    look_ahead_layers: list[int] | tuple[int, ...] = (10,)
    look_backward_layers: list[int] | tuple[int, ...] = (0, 20)
    repeats: int = 2
    num_embeds: int = 8193
    mel_dim: int = 80
    dropout: float | int = 0.1
    enc_emb_dim: int = 192
    enc_dim: int = 128
    enc_channels: list[int] | tuple[int, ...] = (256, 256, 256, 256, 768)
    enc_kernel_sizes: list[int] | tuple[int, ...] = (5, 3, 3, 3, 1)
    enc_dilations: list[int] | tuple[int, ...] = (1, 2, 3, 4, 1)
    enc_attention_channels: int = 64
    enc_res2net_scale: int = 2
    enc_se_channels: int = 64


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniBigVGANConfig(PreTrainedConfig):

    model_type = "qwen2_5_omni_bigvgan"

    mel_dim: int = 80
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: list | tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_rates: list[int] | tuple[int, ...] = (5, 3, 2, 2, 2, 2)
    upsample_kernel_sizes: list[int] | tuple[int, ...] = (11, 7, 4, 4, 4, 4)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniToken2WavConfig(PreTrainedConfig):

    model_type = "qwen2_5_omni_token2wav"
    sub_configs = {
        "dit_config": Qwen2_5OmniDiTConfig,
        "bigvgan_config": Qwen2_5OmniBigVGANConfig,
    }

    dit_config: dict | PreTrainedConfig | None = None
    bigvgan_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.dit_config is None:
            self.dit_config = Qwen2_5OmniDiTConfig()
        elif isinstance(self.dit_config, dict):
            self.dit_config = Qwen2_5OmniDiTConfig(**self.dit_config)

        if self.bigvgan_config is None:
            self.bigvgan_config = Qwen2_5OmniBigVGANConfig()
        elif isinstance(self.bigvgan_config, dict):
            self.bigvgan_config = Qwen2_5OmniBigVGANConfig(**self.bigvgan_config)

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen2.5-Omni-7B")
@strict
class Qwen2_5OmniConfig(PreTrainedConfig):

    model_type = "qwen2_5_omni"
    sub_configs = {
        "thinker_config": Qwen2_5OmniThinkerConfig,
        "talker_config": Qwen2_5OmniTalkerConfig,
        "token2wav_config": Qwen2_5OmniToken2WavConfig,
    }

    thinker_config: dict | PreTrainedConfig | None = None
    talker_config: dict | PreTrainedConfig | None = None
    token2wav_config: dict | PreTrainedConfig | None = None
    enable_audio_output: bool = True

    def __post_init__(self, **kwargs):
        if self.thinker_config is None:
            self.thinker_config = Qwen2_5OmniThinkerConfig()
            logger.info("thinker_config is None. Initializing thinker model with default values")
        elif isinstance(self.thinker_config, dict):
            self.thinker_config = Qwen2_5OmniThinkerConfig(**self.thinker_config)

        if self.talker_config is None:
            self.talker_config = Qwen2_5OmniTalkerConfig()
            logger.info("talker_config is None. Initializing talker model with default values")
        elif isinstance(self.talker_config, dict):
            self.talker_config = Qwen2_5OmniTalkerConfig(**self.talker_config)

        if self.token2wav_config is None:
            self.token2wav_config = Qwen2_5OmniToken2WavConfig()
            logger.info("token2wav_config is None. Initializing token2wav model with default values")
        elif isinstance(self.token2wav_config, dict):
            self.token2wav_config = Qwen2_5OmniToken2WavConfig(**self.token2wav_config)

        super().__post_init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        """
        Returns the config that is meant to be used with text IO. On most models, it is the original config instance
        itself. On specific composite models, it is under a set of valid names.

        Args:
            decoder (`Optional[bool]`, *optional*, defaults to `False`):
                If set to `True`, then only search for decoder config names.
        """
        return self.thinker_config.get_text_config(*args, **kwargs)


__all__ = [
    "Qwen2_5OmniConfig",
    "Qwen2_5OmniThinkerConfig",
    "Qwen2_5OmniTalkerConfig",
    "Qwen2_5OmniToken2WavConfig",
    "Qwen2_5OmniAudioEncoderConfig",
    "Qwen2_5OmniBigVGANConfig",
    "Qwen2_5OmniDiTConfig",
    "Qwen2_5OmniTextConfig",
    "Qwen2_5OmniVisionEncoderConfig",
]
