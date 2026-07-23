
import torch
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring
from ...utils.generic import TransformersKwargs, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import CONFIG_MAPPING, AutoConfig
from ..sam2.configuration_sam2 import Sam2Config, Sam2MaskDecoderConfig, Sam2PromptEncoderConfig
from ..sam2.modeling_sam2 import (
    Sam2Attention,
    Sam2FeedForward,
    Sam2LayerNorm,
    Sam2Model,
    Sam2PreTrainedModel,
    Sam2TwoWayAttentionBlock,
    Sam2VisionEncoderOutput,
    Sam2VisionModel,
)


@auto_docstring(checkpoint="yonigozlan/EdgeTAM-hf")
@strict
class EdgeTamVisionConfig(PreTrainedConfig):

    base_config_key = "vision_config"
    model_type = "edgetam_vision_model"
    sub_configs = {
        "backbone_config": AutoConfig,
    }

    backbone_config: dict | PreTrainedConfig | None = None
    backbone_channel_list: list[int] | None = None
    backbone_feature_sizes: list | None = None
    fpn_hidden_size: int = 256
    fpn_kernel_size: int = 1
    fpn_stride: int = 1
    fpn_padding: int = 0
    fpn_top_down_levels: list[int] | None = None
    num_feature_levels: int = 3
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.backbone_channel_list = (
            [384, 192, 96, 48] if self.backbone_channel_list is None else self.backbone_channel_list
        )
        self.backbone_feature_sizes = (
            [[256, 256], [128, 128], [64, 64]] if self.backbone_feature_sizes is None else self.backbone_feature_sizes
        )
        self.fpn_top_down_levels = [2, 3] if self.fpn_top_down_levels is None else self.fpn_top_down_levels

        if isinstance(self.backbone_config, dict):
            self.backbone_config["model_type"] = self.backbone_config.get("model_type", "timm_wrapper")
            self.backbone_config = CONFIG_MAPPING[self.backbone_config["model_type"]](**self.backbone_config)
        elif self.backbone_config is None:
            self.backbone_config = AutoConfig.from_pretrained(
                "timm/repvit_m1.dist_in1k",
                model_args={"in_chans": 3, "features_only": True, "out_indices": [0, 1, 2, 3]},
            )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="yonigozlan/EdgeTAM-hf")
@strict
class EdgeTamPromptEncoderConfig(Sam2PromptEncoderConfig):
    pass


@auto_docstring(checkpoint="yonigozlan/EdgeTAM-hf")
@strict
class EdgeTamMaskDecoderConfig(Sam2MaskDecoderConfig):
    pass


@auto_docstring(checkpoint="yonigozlan/EdgeTAM-hf")
@strict
class EdgeTamConfig(Sam2Config):

    pass


class EdgeTamLayerNorm(Sam2LayerNorm):
    pass


class EdgeTamVisionEncoderOutput(Sam2VisionEncoderOutput):
    pass


class EdgeTamAttention(Sam2Attention):
    pass


class EdgeTamTwoWayAttentionBlock(Sam2TwoWayAttentionBlock):
    pass


class EdgeTamFeedForward(Sam2FeedForward):
    pass


@auto_docstring
class EdgeTamPreTrainedModel(Sam2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = None

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, EdgeTamModel):
            if module.no_memory_embedding is not None:
                init.zeros_(module.no_memory_embedding)
        elif hasattr(module, "positional_embedding"):
            init.normal_(module.positional_embedding, std=module.scale)


@auto_docstring(
    custom_intro="""
    The vision model from EdgeTAM without any head or projection on top.
    """
)
class EdgeTamVisionModel(Sam2VisionModel):
    config_class = EdgeTamVisionConfig
    main_input_name = "pixel_values"
    _can_record_outputs = {}

    def get_input_embeddings(self):
        raise NotImplementedError("Can't get input embeddings from timm wrapper model")

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | EdgeTamVisionEncoderOutput:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        backbone_output = self.backbone(pixel_values, **kwargs)
        intermediate_hidden_states = backbone_output.last_hidden_state
        intermediate_hidden_states = [hidden_state.permute(0, 2, 3, 1) for hidden_state in intermediate_hidden_states]

        fpn_hidden_states, fpn_position_encoding = self.neck(intermediate_hidden_states)
        fpn_hidden_states = fpn_hidden_states[-self.num_feature_levels :][::-1]
        fpn_position_encoding = fpn_position_encoding[-self.num_feature_levels :][::-1]

        return EdgeTamVisionEncoderOutput(
            last_hidden_state=intermediate_hidden_states[-1],
            fpn_hidden_states=fpn_hidden_states,
            fpn_position_encoding=fpn_position_encoding,
            hidden_states=backbone_output.hidden_states,
        )


class EdgeTamModel(Sam2Model):
    _keys_to_ignore_on_load_unexpected = [
        r"^memory_.*",
        r"^mask_downsample.*",
        r"spatial_perceiver.*",
        r"^object_pointer_proj.*",
        r"^temporal_positional_encoding_projection_layer.*",
        "no_memory_positional_encoding",
        "no_object_pointer",
        "occlusion_spatial_embedding_parameter",
    ]

    def get_input_embeddings(self):
        raise NotImplementedError("Can't get input embeddings from timm wrapper model")


__all__ = [
    "EdgeTamModel",
    "EdgeTamVisionModel",
    "EdgeTamPreTrainedModel",
    "EdgeTamConfig",
    "EdgeTamVisionConfig",
    "EdgeTamPromptEncoderConfig",
    "EdgeTamMaskDecoderConfig",
]
