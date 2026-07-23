

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/git-base")
@strict
class GitVisionConfig(PreTrainedConfig):

    model_type = "git_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="microsoft/git-base")
@strict
class GitConfig(PreTrainedConfig):

    model_type = "git"
    sub_configs = {"vision_config": GitVisionConfig}

    vision_config: dict | GitVisionConfig | None = None
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 1024
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    use_cache: bool = True
    tie_word_embeddings: bool = False
    bos_token_id: int | None = 101
    eos_token_id: int | list[int] | None = 102
    num_image_with_embedding: int | None = None

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = GitVisionConfig()
            logger.info("vision_config is None. initializing the GitVisionConfig with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = GitVisionConfig(**self.vision_config)
        super().__post_init__(**kwargs)


__all__ = ["GitConfig", "GitVisionConfig"]
