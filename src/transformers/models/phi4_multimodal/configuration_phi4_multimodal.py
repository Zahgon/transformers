
import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/Phi-4-multimodal-instruct")
@strict
class Phi4MultimodalVisionConfig(PreTrainedConfig):

    model_type = "phi4_multimodal_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 448
    patch_size: int | list[int] | tuple[int, int] = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    crop_size: int = 448
    image_token_id: int = 200010
    feature_layer: int = -2


@auto_docstring(checkpoint="microsoft/Phi-4-multimodal-instruct")
@strict
class Phi4MultimodalAudioConfig(PreTrainedConfig):

    model_type = "phi4_multimodal_audio"

    hidden_size: int = 1024
    intermediate_size: int = 1536
    num_blocks: int = 24
    num_attention_heads: int = 16
    activation: str = "swish"
    chunk_size: int = -1
    left_chunk: int = 18
    dropout_rate: float | int = 0.0
    ext_pw_out_channel: int = 1024
    depthwise_separable_out_channel: int = 1024
    depthwise_multiplier: int = 1
    kernel_size: int = 3
    conv_activation: str = "swish"
    input_size: int = 80
    conv_glu_type: str = "swish"
    time_reduction: int = 8
    bias_max_distance: int = 1000
    bias_symmetric: bool = False
    nemo_activation: str = "relu"
    nemo_conv_channels: int = 1024
    downsample_rate: int = 1
    initializer_range: float = 0.02
    audio_token_id: int = 200011
    feature_layer: int = -2

    def __post_init__(self, **kwargs):
        nemo_final_size = self.input_size
        for _ in range(int(math.log2(self.time_reduction))):
            nemo_final_size = math.floor((nemo_final_size - 1) / 2 + 1)
        self.nemo_final_size = nemo_final_size
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="microsoft/Phi-4-multimodal-instruct")
@strict
class Phi4MultimodalConfig(PreTrainedConfig):

    model_type = "phi4_multimodal"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.qkv_proj": "colwise_gather_output",  # we need to replicate here due to the slicing of qkv
        "layers.*.self_attn.o_proj": "rowwise_split_input",  # input is replicated due to the slicing of qkv
        "layers.*.mlp.gate_up_proj": "colwise_gather_output",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_split_input",  # input is replicated due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 200064
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    resid_pdrop: float | int = 0.0
    embd_pdrop: float | int = 0.0
    attention_dropout: float | int = 0.0
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int | None = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    bos_token_id: int | None = 199999
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = 199999
    sliding_window: int | None = None

    sub_configs = {"audio_config": Phi4MultimodalAudioConfig, "vision_config": Phi4MultimodalVisionConfig}
    vision_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        self.eos_token_id = self.eos_token_id or [199999, 200020]
        if isinstance(self.vision_config, dict):
            self.vision_config = Phi4MultimodalVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Phi4MultimodalVisionConfig()

        if isinstance(self.audio_config, dict):
            self.audio_config = Phi4MultimodalAudioConfig(**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = Phi4MultimodalAudioConfig()
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(
        self, default_theta: int | float = 10_000.0, ignore_keys: set | None = None, **kwargs
    ):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", default_theta))
        self.rope_parameters.setdefault("partial_rotary_factor", kwargs.get("partial_rotary_factor", 1.0))
        self.standardize_rope_params()

        rope_parameters_type = self.rope_parameters.get("rope_type", None)
        if rope_parameters_type is not None and rope_parameters_type in ["su", "yarn"]:
            self.rope_parameters["rope_type"] = "longrope"
        return kwargs

    def validate_rope(self):
        """
        Validate the `rope_parameters` configuration.
        """
        super().validate_rope()

        if not isinstance(self.rope_parameters, dict):
            raise ValueError(f"`rope_parameters` must be a dictionary but got {self.rope_parameters}")
        rope_parameters_type = self.rope_parameters.get("rope_type", None)
        rope_parameters_short_factor = self.rope_parameters.get("short_factor", None)
        rope_parameters_long_factor = self.rope_parameters.get("long_factor", None)
        rotary_ndims = int(
            self.hidden_size // self.num_attention_heads * self.rope_parameters["partial_rotary_factor"]
        )
        if rope_parameters_type not in ["default", "longrope"]:
            raise ValueError(f"`rope_parameters`'s type field must be one of ['longrope'], got {rope_parameters_type}")

        if rope_parameters_short_factor is not None:
            if not (
                isinstance(rope_parameters_short_factor, list)
                and all(isinstance(x, (int, float)) for x in rope_parameters_short_factor)
            ):
                raise ValueError(
                    f"`rope_parameters`'s short_factor field must be a list of numbers, got {rope_parameters_short_factor}"
                )
            if not len(rope_parameters_short_factor) == rotary_ndims // 2:
                raise ValueError(
                    f"`rope_parameters`'s short_factor field must have length {rotary_ndims // 2}, got {len(rope_parameters_short_factor)}"
                )

        if rope_parameters_long_factor is not None:
            if not (
                isinstance(rope_parameters_long_factor, list)
                and all(isinstance(x, (int, float)) for x in rope_parameters_long_factor)
            ):
                raise ValueError(
                    f"`rope_parameters`'s long_factor field must be a list of numbers, got {rope_parameters_long_factor}"
                )
            if not len(rope_parameters_long_factor) == rotary_ndims // 2:
                raise ValueError(
                    f"`rope_parameters`'s long_factor field must have length {rotary_ndims // 2}, got {len(rope_parameters_long_factor)}"
                )


__all__ = ["Phi4MultimodalVisionConfig", "Phi4MultimodalAudioConfig", "Phi4MultimodalConfig"]
