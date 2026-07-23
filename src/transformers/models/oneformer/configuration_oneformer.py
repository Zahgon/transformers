
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="shi-labs/oneformer_ade20k_swin_tiny")
@strict
class OneFormerConfig(PreTrainedConfig):

    model_type = "oneformer"
    sub_configs = {"backbone_config": AutoConfig}
    attribute_map = {"hidden_size": "hidden_dim", "num_hidden_layers": "decoder_layers"}

    backbone_config: dict | PreTrainedConfig | None = None
    ignore_value: int = 255
    num_queries: int = 150
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    contrastive_weight: float = 0.5
    contrastive_temperature: float = 0.07
    train_num_points: int = 12544
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    init_std: float = 0.02
    init_xavier_std: float = 1.0
    layer_norm_eps: float = 1e-05
    is_training: bool = False
    use_auxiliary_loss: bool = True
    output_auxiliary_logits: bool = True
    strides: list[int] | tuple[int, ...] = (4, 8, 16, 32)
    task_seq_len: int = 77
    text_encoder_width: int = 256
    text_encoder_context_length: int = 77
    text_encoder_num_layers: int = 6
    text_encoder_vocab_size: int = 49408
    text_encoder_proj_layers: int = 2
    text_encoder_n_ctx: int = 16
    conv_dim: int = 256
    mask_dim: int = 256
    hidden_dim: int = 256
    encoder_feedforward_dim: int = 1024
    norm: str = "GN"
    encoder_layers: int = 6
    decoder_layers: int = 10
    use_task_norm: bool = True
    num_attention_heads: int = 8
    dropout: float | int = 0.1
    dim_feedforward: int = 2048
    pre_norm: bool = False
    enforce_input_proj: bool = False
    query_dec_layers: int = 2
    common_stride: int = 4

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="swin",
            default_config_kwargs={
                "drop_path_rate": 0.3,
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            },
            **kwargs,
        )

        super().__post_init__(**kwargs)


__all__ = ["OneFormerConfig"]
