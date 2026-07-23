from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Qwen/Qwen3-Omni-30B-A3B-Instruct")
@strict
class Qwen3OmniMoeAudioEncoderConfig(PreTrainedConfig):

    model_type = "qwen3_omni_moe_audio_encoder"
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

    n_window: int = 50
    output_dim: int = 3584
    n_window_infer: int = 800
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480


@auto_docstring(checkpoint="Qwen/Qwen3-Omni-30B-A3B-Instruct")
@strict
class Qwen3OmniMoeVisionEncoderConfig(PreTrainedConfig):

    model_type = "qwen3_omni_moe_vision_encoder"
    base_config_key = "vision_config"

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] | tuple[int, ...] = (8, 16, 24)
    initializer_range: float = 0.02


@auto_docstring(checkpoint="Qwen/Qwen3-Omni-30B-A3B-Instruct")
@strict
class Qwen3OmniMoeTextConfig(PreTrainedConfig):

    model_type = "qwen3_omni_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    ignore_keys_at_rope_validation = {"mrope_section", "interleaved", "mrope_interleaved"}

    vocab_size: int = 3584
    hidden_size: int = 2048
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    sliding_window: int | None = None
    attention_dropout: float | int = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 768
    num_experts_per_tok: int = 8
    num_experts: int = 128
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list[int] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen3-Omni-30B-A3B-Instruct")
@strict
class Qwen3OmniMoeThinkerConfig(PreTrainedConfig):

    model_type = "qwen3_omni_moe_thinker"
    attribute_map = {}
    sub_configs = {
        "audio_config": Qwen3OmniMoeAudioEncoderConfig,
        "vision_config": Qwen3OmniMoeVisionEncoderConfig,
        "text_config": Qwen3OmniMoeTextConfig,
    }

    audio_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    position_id_per_seconds: int = 25
    audio_start_token_id: int = 151647
    user_token_id: int = 872
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    audio_token_id: int = 151646
    image_token_id: int = 151655
    video_token_id: int = 151656

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = Qwen3OmniMoeVisionEncoderConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Qwen3OmniMoeVisionEncoderConfig()

        if isinstance(self.audio_config, dict):
            self.audio_config = Qwen3OmniMoeAudioEncoderConfig(**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = Qwen3OmniMoeAudioEncoderConfig()

        if isinstance(self.text_config, dict):
            self.text_config = Qwen3OmniMoeTextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = Qwen3OmniMoeTextConfig()

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen3-Omni-30B-A3B-Instruct")
@strict
class Qwen3OmniMoeTalkerCodePredictorConfig(PreTrainedConfig):

    model_type = "qwen3_omni_moe_talker_code_predictor"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
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

    vocab_size: int = 2048
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    sliding_window: int | None = None
    max_window_layers: int = 28
    layer_types: list[str] | None = None
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    num_code_groups: int = 32

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window
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


@auto_docstring(checkpoint="Qwen/Qwen3-Omni-30B-A3B-Instruct")
@strict
class Qwen3OmniMoeTalkerTextConfig(PreTrainedConfig):

    model_type = "qwen3_omni_moe_talker_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_experts": "num_local_experts",
    }

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 3072
    hidden_size: int = 1024
    intermediate_size: int = 2048
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    sliding_window: int | None = None
    attention_dropout: float | int = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 384
    num_experts_per_tok: int = 8
    num_experts: int = 128
    norm_topk_prob: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list[int] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen3-Omni-30B-A3B-Instruct")
@strict
class Qwen3OmniMoeTalkerConfig(PreTrainedConfig):

    sub_configs = {
        "code_predictor_config": Qwen3OmniMoeTalkerCodePredictorConfig,
        "text_config": Qwen3OmniMoeTalkerTextConfig,
    }

    code_predictor_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    num_code_groups: int = 32
    thinker_hidden_size: int = 2048
    codec_eos_token_id: int = 4198
    accept_hidden_layer: int = 18
    codec_nothink_id: int = 4203
    codec_think_bos_id: int = 4204
    codec_think_eos_id: int = 4205
    codec_pad_id: int = 4196
    codec_bos_id: int = 4197
    audio_token_id: int = 151646
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    position_id_per_seconds: int = 25
    audio_start_token_id: int = 151669
    speaker_id: dict | None = None
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.code_predictor_config is None:
            self.code_predictor_config = {}
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig()
            logger.info("code_predictor_config is None. Initializing code_predictor_config model with default values")
        else:
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig(**self.code_predictor_config)

        if self.text_config is None:
            self.text_config = {}
            self.text_config = Qwen3OmniMoeTalkerTextConfig()
            logger.info("talker text_config is None. Initializing talker text model with default values")
        else:
            self.text_config = Qwen3OmniMoeTalkerTextConfig(**self.text_config)
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen3-Omni-30B-A3B-Instruct")
@strict
class Qwen3OmniMoeCode2WavConfig(PreTrainedConfig):

    codebook_size: int = 2048
    hidden_size: int = 1024
    max_position_embeddings: int = 8000
    rope_parameters: RopeParameters | dict | None = None
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    attention_bias: bool = False
    sliding_window: int = 72
    intermediate_size: int = 3072
    hidden_act: str = "silu"
    layer_scale_initial_scale: float = 0.01
    rms_norm_eps: float = 1e-5
    num_hidden_layers: int = 8
    num_quantizers: int = 16
    upsample_rates: list[int] | tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: list[int] | tuple[int, ...] = (2, 2)
    decoder_dim: int = 1536
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02

    @property
    def layer_types(self):
        pass


@auto_docstring(checkpoint="Qwen/Qwen3-Omni-30B-A3B-Instruct")
@strict
class Qwen3OmniMoeConfig(PreTrainedConfig):

    model_type = "qwen3_omni_moe"
    sub_configs = {
        "thinker_config": Qwen3OmniMoeThinkerConfig,
        "talker_config": Qwen3OmniMoeTalkerConfig,
        "code2wav_config": Qwen3OmniMoeCode2WavConfig,
    }

    thinker_config: dict | PreTrainedConfig | None = None
    talker_config: dict | PreTrainedConfig | None = None
    code2wav_config: dict | PreTrainedConfig | None = None
    enable_audio_output: bool = True
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    tts_pad_token_id: int = 151671
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    system_token_id: int = 8948
    user_token_id: int = 872
    assistant_token_id: int = 77091
    initializer_range: float | None = None

    def __post_init__(self, **kwargs):
        if self.thinker_config is None:
            self.thinker_config = Qwen3OmniMoeThinkerConfig()
            logger.info("thinker_config is None. Initializing thinker model with default values")
        elif isinstance(self.thinker_config, dict):
            self.thinker_config = Qwen3OmniMoeThinkerConfig(**self.thinker_config)

        if self.talker_config is None:
            self.talker_config = Qwen3OmniMoeTalkerConfig()
            logger.info("talker_config is None. Initializing talker model with default values")
        elif isinstance(self.talker_config, dict):
            self.talker_config = Qwen3OmniMoeTalkerConfig(**self.talker_config)

        if self.code2wav_config is None:
            self.code2wav_config = Qwen3OmniMoeCode2WavConfig()
            logger.info("code2wav_config is None. Initializing code2wav_config model with default values")
        elif isinstance(self.code2wav_config, dict):
            self.code2wav_config = Qwen3OmniMoeCode2WavConfig(**self.code2wav_config)

        if self.initializer_range is None:
            self.initializer_range = self.thinker_config.initializer_range

        super().__post_init__(**kwargs)

    def get_text_config(self, decoder=False) -> "PreTrainedConfig":
        """
        Returns the config that is meant to be used with text IO. On most models, it is the original config instance
        itself. On specific composite models, it is under a set of valid names.

        Args:
            decoder (`Optional[bool]`, *optional*, defaults to `False`):
                If set to `True`, then only search for decoder config names.
        """
        return self.thinker_config.get_text_config()


__all__ = [
    "Qwen3OmniMoeAudioEncoderConfig",
    "Qwen3OmniMoeConfig",
    "Qwen3OmniMoeThinkerConfig",
    "Qwen3OmniMoeTalkerConfig",
    "Qwen3OmniMoeTalkerCodePredictorConfig",
    "Qwen3OmniMoeTalkerTextConfig",
    "Qwen3OmniMoeTextConfig",
    "Qwen3OmniMoeVisionEncoderConfig",
]
