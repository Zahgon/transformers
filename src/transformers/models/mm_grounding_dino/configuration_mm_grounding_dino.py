from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det")
@strict
class MMGroundingDinoConfig(PreTrainedConfig):

    model_type = "mm-grounding-dino"
    sub_configs = {"backbone_config": AutoConfig, "text_config": AutoConfig}
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    backbone_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    num_queries: int = 900
    encoder_layers: int = 6
    encoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 8
    decoder_layers: int = 6
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 8
    is_encoder_decoder: bool = True
    activation_function: str = "relu"
    d_model: int = 256
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    auxiliary_loss: bool = False
    position_embedding_type: str = "sine"
    num_feature_levels: int = 4
    encoder_n_points: int = 4
    decoder_n_points: int = 4
    two_stage: bool = True
    class_cost: float = 1.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0
    bbox_loss_coefficient: float = 5.0
    giou_loss_coefficient: float = 2.0
    focal_alpha: float = 0.25
    disable_custom_kernels: bool = False
    max_text_len: int = 256
    text_enhancer_dropout: float | int = 0.0
    fusion_droppath: float | int = 0.1
    fusion_dropout: float | int = 0.0
    embedding_init_target: bool = True
    query_dim: int = 4
    positional_embedding_temperature: int = 20
    init_std: float = 0.02
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="swin",
            default_config_kwargs={"out_indices": [2, 3, 4]},
            **kwargs,
        )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "bert")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            logger.info("text_config is None. Initializing the text config with default values (`BertConfig`).")
            self.text_config = CONFIG_MAPPING["bert"]()

        super().__post_init__(**kwargs)


__all__ = ["MMGroundingDinoConfig"]
