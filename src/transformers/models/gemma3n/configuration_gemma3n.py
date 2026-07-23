from collections.abc import Sequence
from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, is_timm_available, logging, requires_backends


if is_timm_available():
    from timm.data import ImageNetInfo, infer_imagenet_subset

logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/gemma-3n-E4B")
@strict
class Gemma3nTextConfig(PreTrainedConfig):

    model_type = "gemma3n_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.v_norm": "replicated_with_grad_allreduce",
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

    vocab_size: int = 262_400
    hidden_size: int = 2048
    intermediate_size: int | list[int] = 16_384
    num_hidden_layers: int = 35
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 32_768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    sliding_window: int = 512
    layer_types: list[str] | None = None
    final_logit_softcapping: float = 30.0
    default_theta = {"global": 1_000_000.0, "local": 10_000.0}
    vocab_size_per_layer_input: int = 262_144
    hidden_size_per_layer_input: int = 256
    altup_active_idx: int = 0
    altup_coef_clip: float = 120.0
    altup_correct_scale: bool = True
    altup_num_inputs: int = 4
    num_kv_shared_layers: int = 15
    laurel_rank: int = 64
    activation_sparsity_pattern: float | list[float] | None = None

    def __post_init__(self, **kwargs):
        if (
            isinstance(self.intermediate_size, Sequence)
            and (intsize_len := len(self.intermediate_size)) != self.num_hidden_layers
        ):
            raise ValueError(
                "intermediate_size must have an explicit intermediate size for every layer or one for all layers. "
                f"Expected {self.num_hidden_layers} values but got {intsize_len}."
            )
        elif not isinstance(self.intermediate_size, Sequence):
            self.intermediate_size = [self.intermediate_size] * self.num_hidden_layers

        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 5 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]

        if self.activation_sparsity_pattern is None:
            num_sparse_layers = 10 if self.num_hidden_layers > 10 else 0
            self.activation_sparsity_pattern = [0.95] * num_sparse_layers + [0.0] * (
                self.num_hidden_layers - num_sparse_layers
            )

        if (len_asp := len(self.activation_sparsity_pattern)) != self.num_hidden_layers:
            raise ValueError(
                "activation_sparsity_pattern must have an explicit activation sparsity value for every layer."
                f"Expected {self.num_hidden_layers} values but got {len_asp}."
            )

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)

        if self.rope_parameters.get("full_attention") is None:
            self.rope_parameters["full_attention"] = {"rope_type": "default"}
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_theta", self.default_theta["global"])
        )
        if self.rope_parameters.get("sliding_attention") is None:
            self.rope_parameters["sliding_attention"] = {"rope_type": "default"}
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_local_base_freq", self.default_theta["local"])
        )

        self.standardize_rope_params()
        return kwargs


@auto_docstring(checkpoint="google/gemma-3n-E4B")
@strict
class Gemma3nAudioConfig(PreTrainedConfig):

    model_type = "gemma3n_audio"

    vocab_size: int = 128
    vocab_offset: int = 262_144 + 128  # text vocab size + vision vocab size
    input_feat_size: int = 128
    hidden_size: int = 1536
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 10_000_000_000.0
    conf_attention_chunk_size: int = 12
    conf_attention_context_left: int = 13
    conf_attention_context_right: int = 0
    conf_attention_logit_cap: float = 50.0
    conf_num_attention_heads: int = 8
    conf_num_hidden_layers: int = 12
    conf_conv_kernel_size: int = 5
    conf_reduction_factor: int = 4
    conf_residual_weight: float = 0.5
    sscp_conv_channel_size: list[int] | tuple[int, int] = (128, 32)
    sscp_conv_group_norm_eps: float = 1e-3
    sscp_conv_kernel_size: list | tuple[tuple[int, int], tuple[int, int]] = (
        (3, 3),
        (3, 3),
    )
    sscp_conv_stride_size: list | tuple[tuple[int, int], tuple[int, int]] = (
        (2, 2),
        (2, 2),
    )


@auto_docstring(checkpoint="google/gemma-3n-E4B")
@strict
class Gemma3nVisionConfig(PreTrainedConfig):

    model_type = "gemma3n_vision"
    architecture: str = "mobilenetv5_300m_enc"

    initializer_range: float = 0.02
    do_pooling: bool = False
    model_args: dict | None = None
    hidden_size: int = 2048
    vocab_size: int = 128
    vocab_offset: int = 262_144
    rms_norm_eps: float = 1e-06

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):
        config_dict = config_dict.copy()

        label_names = config_dict.get("label_names")
        is_custom_model = "num_labels" in kwargs or "id2label" in kwargs

        if label_names is None and not is_custom_model:
            requires_backends(cls, ["timm"])
            imagenet_subset = infer_imagenet_subset(config_dict)
            if imagenet_subset:
                dataset_info = ImageNetInfo(imagenet_subset)
                synsets = dataset_info.label_names()
                label_descriptions = dataset_info.label_descriptions(as_dict=True)
                label_names = [label_descriptions[synset] for synset in synsets]

        if label_names is not None and not is_custom_model:
            kwargs["id2label"] = dict(enumerate(label_names))

            if len(set(label_names)) == len(label_names):
                kwargs["label2id"] = {name: i for i, name in enumerate(label_names)}
            else:
                kwargs["label2id"] = None

        num_labels_in_kwargs = kwargs.pop("num_labels", None)
        num_labels_in_dict = config_dict.pop("num_classes", None)

        kwargs["num_labels"] = num_labels_in_kwargs or num_labels_in_dict

        if "pretrained_cfg" in config_dict and "num_classes" in config_dict["pretrained_cfg"]:
            config_dict["pretrained_cfg"].pop("num_classes", None)

        return super().from_dict(config_dict, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output.setdefault("num_classes", self.num_labels)
        output.setdefault("label_names", list(self.id2label.values()))
        output.pop("id2label", None)
        output.pop("label2id", None)
        return output


@auto_docstring(checkpoint="google/gemma-3n-E4B")
@strict
class Gemma3nConfig(PreTrainedConfig):

    model_type = "gemma3n"
    sub_configs = {
        "text_config": Gemma3nTextConfig,
        "vision_config": Gemma3nVisionConfig,
        "audio_config": Gemma3nAudioConfig,
    }

    text_config: Gemma3nTextConfig | dict[str, Any] | None = None
    vision_config: Gemma3nVisionConfig | dict[str, Any] | None = None
    audio_config: Gemma3nAudioConfig | dict[str, Any] | None = None
    audio_soft_tokens_per_image: int | None = 188
    vision_soft_tokens_per_image: int | None = 256
    boi_token_id: int | None = 255_999
    eoi_token_id: int | None = 262_144
    image_token_id: int | None = 262_145
    boa_token_id: int | None = 256_000
    eoa_token_id: int | None = 262_272
    audio_token_id: int | None = 262_273
    initializer_range: float | None = 0.02
    tie_word_embeddings: bool | None = True
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Gemma3nTextConfig()
            logger.info("text_config is None, using default Gemma3nTextConfig text config.")
        elif isinstance(self.text_config, dict):
            self.text_config = Gemma3nTextConfig(**self.text_config)

        if isinstance(self.vision_config, dict):
            self.vision_config = Gemma3nVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Gemma3nVisionConfig()
            logger.info("vision_config is None, using default Gemma3nVisionConfig vision config.")

        if isinstance(self.audio_config, dict):
            self.audio_config = Gemma3nAudioConfig(**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = Gemma3nAudioConfig()
            logger.info("audio_config is None. Using default Gemma3nAudioConfig.")

        super().__post_init__(**kwargs)


__all__ = ["Gemma3nAudioConfig", "Gemma3nConfig", "Gemma3nTextConfig", "Gemma3nVisionConfig"]
