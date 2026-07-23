
import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...backbone_utils import (
    consolidate_backbone_kwargs_to_config,
)
from ...configuration_utils import PreTrainedConfig
from ...utils import (
    auto_docstring,
    logging,
)
from ..pp_lcnet.modeling_pp_lcnet import PPLCNetDepthwiseSeparableConvLayer
from ..pp_ocrv5_mobile_det.configuration_pp_ocrv5_mobile_det import PPOCRV5MobileDetConfig
from ..pp_ocrv5_mobile_det.modeling_pp_ocrv5_mobile_det import (
    PPOCRV5MobileDetHead,
    PPOCRV5MobileDetNeck,
    PPOCRV5MobileDetResidualSqueezeExcitationLayer,
    PPOCRV5MobileDetSqueezeExcitationModule,
)
from ..pp_ocrv5_server_det.modeling_pp_ocrv5_server_det import (
    PPOCRV5ServerDetForObjectDetection,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv6_small_det_safetensors")
@strict
class PPOCRV6SmallDetConfig(PPOCRV5MobileDetConfig):

    dilated_kernel_size: int = 7

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet_v4",
            **kwargs,
        )

        self.id2label = {0: "text"} if self.id2label is None else self.id2label
        PreTrainedConfig.__post_init__(**kwargs)


class PPOCRV6SmallDetHead(PPOCRV5MobileDetHead):
    pass


class PPOCRV6SmallDetSqueezeExcitationModule(PPOCRV5MobileDetSqueezeExcitationModule):
    pass


class PPOCRV6SmallDetDepthwiseSeparableConvLayer(PPLCNetDepthwiseSeparableConvLayer):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        reduction,
    ):
        super().__init__()
        self.depthwise_convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=True,
        )
        self.squeeze_excitation_module = PPOCRV6SmallDetSqueezeExcitationModule(out_channels // 4, reduction)
        self.pointwise_convolution = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels // 4,
            kernel_size=1,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.depthwise_convolution(hidden_states)
        hidden_states = self.pointwise_convolution(hidden_states)
        hidden_states = hidden_states + self.squeeze_excitation_module(hidden_states)

        return hidden_states


class PPOCRV6SmallDetResidualSqueezeExcitationLayer(PPOCRV5MobileDetResidualSqueezeExcitationLayer):
    pass


class PPOCRV6SmallDetNeck(PPOCRV5MobileDetNeck):

    def __init__(self, config: PPOCRV6SmallDetConfig):
        nn.Module.__init__(self)
        self.interpolate_mode = config.interpolate_mode

        self.insert_conv = nn.ModuleList()
        self.input_conv = nn.ModuleList()
        for i in range(len(config.layer_list_out_channels)):
            self.insert_conv.append(
                PPOCRV6SmallDetResidualSqueezeExcitationLayer(
                    config.layer_list_out_channels[i],
                    config.neck_out_channels,
                    1,
                    config.reduction,
                )
            )
            self.input_conv.append(
                PPOCRV6SmallDetDepthwiseSeparableConvLayer(
                    config.neck_out_channels, config.neck_out_channels, config.dilated_kernel_size, config.reduction
                )
            )


@auto_docstring(custom_intro="PPOCR6SmallRec model for text recognition tasks.")
class PPOCRV6SmallDetForObjectDetection(PPOCRV5ServerDetForObjectDetection):
    pass


__all__ = [
    "PPOCRV6SmallDetForObjectDetection",
    "PPOCRV6SmallDetConfig",
    "PPOCRV6SmallDetModel",  # noqa: F822
    "PPOCRV6SmallDetPreTrainedModel",  # noqa: F822
]
