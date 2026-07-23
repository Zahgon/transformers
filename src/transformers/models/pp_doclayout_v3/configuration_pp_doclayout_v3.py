
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="PaddlePaddle/PP-DocLayoutV3_safetensors")
@strict
class PPDocLayoutV3Config(PreTrainedConfig):

    model_type = "pp_doclayout_v3"
    sub_configs = {"backbone_config": AutoConfig}

    layer_types = ("basic", "bottleneck")
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    initializer_range: float = 0.01
    initializer_bias_prior_prob: float | None = None
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    backbone_config: dict | PreTrainedConfig | None = None
    freeze_backbone_batch_norms: bool = True
    encoder_hidden_dim: int = 256
    encoder_in_channels: list[int] | tuple[int, ...] = (512, 1024, 2048)
    feat_strides: list[int] | tuple[int, ...] = (8, 16, 32)
    encoder_layers: int = 1
    encoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 8
    dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    encode_proj_layers: list[int] | tuple[int, ...] = (2,)
    positional_encoding_temperature: int = 10000
    encoder_activation_function: str = "gelu"
    activation_function: str = "silu"
    eval_size: int | None = None
    normalize_before: bool = False
    hidden_expansion: float = 1.0
    mask_feature_channels: list[int] | tuple[int, ...] = (64, 64)
    x4_feat_dim: int = 128
    d_model: int = 256
    num_prototypes: int = 32
    label_noise_ratio: float = 0.4
    box_noise_scale: float = 0.4
    mask_enhanced: bool = True
    num_queries: int = 300
    decoder_in_channels: list[int] | tuple[int, ...] = (256, 256, 256)
    decoder_ffn_dim: int = 1024
    num_feature_levels: int = 3
    decoder_n_points: int = 4
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_activation_function: str = "relu"
    attention_dropout: float | int = 0.0
    num_denoising: int = 100
    learn_initial_query: bool = False
    anchor_image_size: int | None = None
    disable_custom_kernels: bool = True
    is_encoder_decoder: bool = True
    global_pointer_head_size: int = 64
    gp_dropout_value: float | int = 0.1

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="hgnet_v2",
            default_config_kwargs={
                "arch": "L",
                "return_idx": [0, 1, 2, 3],
                "freeze_stem_only": True,
                "freeze_at": 0,
                "freeze_norm": True,
                "lr_mult_list": [0, 0.05, 0.05, 0.05, 0.05],
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
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


__all__ = ["PPDocLayoutV3Config"]
