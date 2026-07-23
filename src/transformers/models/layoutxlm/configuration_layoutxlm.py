

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, is_detectron2_available


if is_detectron2_available():
    import detectron2


@auto_docstring(checkpoint="microsoft/layoutxlm-base")
@strict
class LayoutXLMConfig(PreTrainedConfig):

    model_type = "layoutxlm"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    max_2d_position_embeddings: int = 1024
    max_rel_pos: int = 128
    rel_pos_bins: int = 32
    fast_qkv: bool = True
    max_rel_2d_pos: int = 256
    rel_2d_pos_bins: int = 64
    convert_sync_batchnorm: bool = True
    image_feature_pool_shape: list[int] | tuple[int, ...] = (7, 7, 256)
    coordinate_size: int = 128
    shape_size: int = 128
    has_relative_attention_bias: bool = True
    has_spatial_attention_bias: bool = True
    has_visual_segment_embedding: bool = False
    detectron2_config_args: dict | None = None

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.detectron2_config_args = (
            self.detectron2_config_args
            if self.detectron2_config_args is not None
            else self.get_default_detectron2_config()
        )

    @classmethod
    def get_default_detectron2_config(cls):
        return {
            "MODEL.MASK_ON": True,
            "MODEL.PIXEL_STD": [57.375, 57.120, 58.395],
            "MODEL.BACKBONE.NAME": "build_resnet_fpn_backbone",
            "MODEL.FPN.IN_FEATURES": ["res2", "res3", "res4", "res5"],
            "MODEL.ANCHOR_GENERATOR.SIZES": [[32], [64], [128], [256], [512]],
            "MODEL.RPN.IN_FEATURES": ["p2", "p3", "p4", "p5", "p6"],
            "MODEL.RPN.PRE_NMS_TOPK_TRAIN": 2000,
            "MODEL.RPN.PRE_NMS_TOPK_TEST": 1000,
            "MODEL.RPN.POST_NMS_TOPK_TRAIN": 1000,
            "MODEL.POST_NMS_TOPK_TEST": 1000,
            "MODEL.ROI_HEADS.NAME": "StandardROIHeads",
            "MODEL.ROI_HEADS.NUM_CLASSES": 5,
            "MODEL.ROI_HEADS.IN_FEATURES": ["p2", "p3", "p4", "p5"],
            "MODEL.ROI_BOX_HEAD.NAME": "FastRCNNConvFCHead",
            "MODEL.ROI_BOX_HEAD.NUM_FC": 2,
            "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION": 14,
            "MODEL.ROI_MASK_HEAD.NAME": "MaskRCNNConvUpsampleHead",
            "MODEL.ROI_MASK_HEAD.NUM_CONV": 4,
            "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION": 7,
            "MODEL.RESNETS.DEPTH": 101,
            "MODEL.RESNETS.SIZES": [[32], [64], [128], [256], [512]],
            "MODEL.RESNETS.ASPECT_RATIOS": [[0.5, 1.0, 2.0]],
            "MODEL.RESNETS.OUT_FEATURES": ["res2", "res3", "res4", "res5"],
            "MODEL.RESNETS.NUM_GROUPS": 32,
            "MODEL.RESNETS.WIDTH_PER_GROUP": 8,
            "MODEL.RESNETS.STRIDE_IN_1X1": False,
        }

    def get_detectron2_config(self):
        detectron2_config = detectron2.config.get_cfg()
        for k, v in self.detectron2_config_args.items():
            attributes = k.split(".")
            to_set = detectron2_config
            for attribute in attributes[:-1]:
                to_set = getattr(to_set, attribute)
            setattr(to_set, attributes[-1], v)

        return detectron2_config


__all__ = ["LayoutXLMConfig"]
