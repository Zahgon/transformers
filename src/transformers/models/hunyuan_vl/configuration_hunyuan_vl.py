
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(
    custom_intro="""
    Vision backbone configuration for the dense-only, image-text HunYuanVL open-source variant.
    """,
    checkpoint="tencent/HunyuanOCR",
)
@strict
class HunYuanVLVisionConfig(PreTrainedConfig):

    model_type = "hunyuan_vl_vision"
    base_config_key = "vision_config"
    attribute_map = {"attention_heads": "num_attention_heads", "layer_norm_eps": "rms_norm_eps"}

    hidden_act: str = "gelu"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    interpolate_mode: str = "bilinear"
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    num_channels: int = 3
    num_hidden_layers: int = 27
    out_hidden_size: int = 4096
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    img_max_token_num: int = 4096
    max_image_size: int = 2048
    min_image_size: int = 512
    max_vit_seq_len: int = 16384
    text_hidden_size: int = 3072

    def __post_init__(self, **kwargs):
        if not self.num_key_value_heads:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(
    custom_intro="""
    Text backbone configuration for the dense-only, image-text HunYuanVL open-source variant.

    Inherits the standard fields from [`HunYuanDenseV1Config`] and declares the canonical field names
    (`pad_token_id`, `head_dim`, `vocab_size`) as the only public attributes. Legacy aliases that some Tencent
    checkpoints persist on disk (`pad_id`, `attention_head_dim`, `org_vocab_size`) are mapped onto those canonical
    fields via `attribute_map`, so the rest of the model only ever needs to read the canonical fields. Legacy RoPE
    payloads persisted as `rope_scaling` / `rope_theta` are normalized by the base configuration class into
    `rope_parameters`.
    """,
    checkpoint="tencent/HunyuanOCR",
)
@strict
class HunYuanVLTextConfig(PreTrainedConfig):

    model_type = "hunyuan_vl_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 290943
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    eod_token_id: int | None = 3
    pretraining_tp: int = 1
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    head_dim: int | None = None
    base_config_key = "text_config"
    ignore_keys_at_rope_validation = {
        "alpha",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
        "mrope_section",
    }

    attribute_map = {
        "pad_id": "pad_token_id",
    }

    sep_token_id: int | None = 4

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)
        rope_parameters = getattr(self, "rope_parameters", None)
        head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        self._validate_mrope_section(rope_parameters, head_dim)

    @staticmethod
    def _normalize_mrope_section_alias(rope_parameters):
        if not rope_parameters:
            return

        legacy_section = rope_parameters.pop("xdrope_section", None)
        if legacy_section is None:
            return

        mrope_section = rope_parameters.get("mrope_section")
        if mrope_section is not None:
            legacy_values = [float(section) for section in legacy_section]
            mrope_values = [float(section) for section in mrope_section]
            if legacy_values != mrope_values:
                raise ValueError(
                    "`rope_parameters` contains both `mrope_section` and legacy `xdrope_section`, but they differ: "
                    f"mrope_section={mrope_section}, xdrope_section={legacy_section}."
                )
            return

        rope_parameters["mrope_section"] = legacy_section

    @staticmethod
    def _validate_mrope_section(rope_parameters, head_dim):
        if not rope_parameters or rope_parameters.get("mrope_section") is None:
            return

        mrope_section = rope_parameters["mrope_section"]
        section_values = [float(section) for section in mrope_section]
        section_ints = [int(section) for section in section_values]
        expected_sum = head_dim // 2
        if not all(value.is_integer() for value in section_values) or sum(section_ints) != expected_sum:
            raise ValueError(
                f"Illegal mrope partition: expected half-head sections summing to {expected_sum}, got {section_ints}"
            )
        rope_parameters["mrope_section"] = section_ints

    def convert_rope_params_to_dict(self, **kwargs):
        kwargs = super().convert_rope_params_to_dict(**kwargs)

        rope_parameters = getattr(self, "rope_parameters", None)
        if not rope_parameters:
            return kwargs

        self._normalize_mrope_section_alias(rope_parameters)
        rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        if rope_type == "xdrope":
            rope_type = "dynamic"
        rope_parameters["rope_type"] = rope_type
        if "type" in rope_parameters:
            rope_parameters["type"] = rope_type
        return kwargs


@auto_docstring(
    custom_intro="""
    Top-level configuration for the open-source HunYuanVL integration.

    This configuration describes the dense-only, image-text-only variant used for OCR and document-understanding style
    workloads. It mirrors the `Qwen2_5_VL` / `Qwen3_VL` family layout: the top-level config simply composes a
    [`HunYuanVLTextConfig`] (text backbone) and a [`HunYuanVLVisionConfig`] (vision tower) plus a few token ids that
    delimit image spans in multimodal prompts.
    """,
    checkpoint="tencent/HunyuanOCR",
)
@strict
class HunYuanVLConfig(PreTrainedConfig):

    model_type = "hunyuan_vl"
    sub_configs = {"vision_config": HunYuanVLVisionConfig, "text_config": HunYuanVLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None

    image_token_id: int = 120120
    tie_word_embeddings: bool = True
    im_start_id: int = 120118
    im_end_id: int = 120119
    im_newline_id: int = 120121

    def __post_init__(self, **kwargs):
        text_keys = set(self.sub_configs["text_config"].__dataclass_fields__) | {"rope_scaling", "rope_theta"}
        text_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in text_keys}

        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**{**self.text_config, **text_kwargs})
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"](**text_kwargs)

        self.vision_config.text_hidden_size = self.text_config.hidden_size

        kwargs.setdefault("tie_word_embeddings", self.text_config.tie_word_embeddings)
        super().__post_init__(**kwargs)


__all__ = ["HunYuanVLConfig", "HunYuanVLVisionConfig", "HunYuanVLTextConfig"]
