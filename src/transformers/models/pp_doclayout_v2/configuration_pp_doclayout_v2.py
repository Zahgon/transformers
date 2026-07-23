
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="PaddlePaddle/PP-DocLayoutV2_safetensors")
@strict
class PPDocLayoutV2ReadingOrderConfig(PreTrainedConfig):

    hidden_size: int = 512
    num_attention_heads: int = 8
    attention_probs_dropout_prob: float | int = 0.1
    has_relative_attention_bias: bool = False
    has_spatial_attention_bias: bool = True
    layer_norm_eps: float = 1e-5
    hidden_dropout_prob: float | int = 0.1
    intermediate_size: int = 2048
    hidden_act: str = "gelu"
    num_hidden_layers: int = 6
    rel_pos_bins: int = 32
    max_rel_pos: int = 128
    rel_2d_pos_bins: int = 64
    max_rel_2d_pos: int = 256
    max_position_embeddings: int = 514
    max_2d_position_embeddings: int = 1024
    type_vocab_size: int = 1
    vocab_size: int = 4
    initializer_range: float = 0.01
    start_token_id: int = 0
    pad_token_id: int | None = 1
    end_token_id: int = 2
    pred_token_id: int = 3
    coordinate_size: int = 171
    shape_size: int = 170
    num_classes: int = 20
    relation_bias_embed_dim: int = 16
    relation_bias_theta: int = 10000
    relation_bias_scale: int = 100
    global_pointer_head_size: int = 64
    gp_dropout_value: float | int = 0.0


@auto_docstring(checkpoint="PaddlePaddle/PP-DocLayoutV2_safetensors")
@strict
class PPDocLayoutV2Config(PreTrainedConfig):

    model_type = "pp_doclayout_v2"
    sub_configs = {"backbone_config": AutoConfig, "reading_order_config": PPDocLayoutV2ReadingOrderConfig}

    layer_types = ("basic", "bottleneck")
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    initializer_range: float = 0.01
    initializer_bias_prior_prob: float | None = None
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    backbone_config: PreTrainedConfig | dict | None = None
    freeze_backbone_batch_norms: bool = True
    encoder_hidden_dim: int = 256
    encoder_in_channels: list[int] | tuple[int, ...] | None = (512, 1024, 2048)
    feat_strides: list[int] | tuple[int, ...] | None = (8, 16, 32)
    encoder_layers: int = 1
    encoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 8
    dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    encode_proj_layers: list[int] | tuple[int, ...] | None = (2,)
    positional_encoding_temperature: int = 10000
    encoder_activation_function: str = "gelu"
    activation_function: str = "silu"
    eval_size: list[int] | None = None
    normalize_before: bool = False
    hidden_expansion: float = 1.0
    d_model: int = 256
    num_queries: int = 300
    decoder_in_channels: list[int] | tuple[int, ...] | None = (256, 256, 256)
    decoder_ffn_dim: int = 1024
    num_feature_levels: int = 3
    decoder_n_points: int = 4
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_activation_function: str = "relu"
    attention_dropout: float | int = 0.0
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    learn_initial_query: bool = False
    anchor_image_size: list[int] | None = None
    disable_custom_kernels: bool = True
    is_encoder_decoder: bool = True
    class_thresholds: list[float] | None = None
    class_order: list[int] | None = None
    reading_order_config: PreTrainedConfig | dict | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.reading_order_config, dict):
            self.reading_order_config = self.sub_configs["reading_order_config"](**self.reading_order_config)
        elif self.reading_order_config is None:
            self.reading_order_config = self.sub_configs["reading_order_config"]()

        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="hgnet_v2",
            default_config_kwargs={
                "arch": "L",
                "return_idx": [1, 2, 3],
                "freeze_stem_only": True,
                "freeze_at": 0,
                "freeze_norm": True,
                "lr_mult_list": [0, 0.05, 0.05, 0.05, 0.05],
                "out_features": ["stage2", "stage3", "stage4"],
            },
            **kwargs,
        )

        self.encoder_in_channels = list(self.encoder_in_channels)
        self.feat_strides = list(self.feat_strides)
        self.encode_proj_layers = list(self.encode_proj_layers)
        self.eval_size = list(self.eval_size) if self.eval_size is not None else None
        self.decoder_in_channels = list(self.decoder_in_channels)
        self.anchor_image_size = list(self.anchor_image_size) if self.anchor_image_size is not None else None

        super().__post_init__(**kwargs)


__all__ = ["PPDocLayoutV2Config"]
