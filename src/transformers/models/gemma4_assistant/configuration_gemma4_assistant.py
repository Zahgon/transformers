
from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..gemma4.configuration_gemma4 import Gemma4TextConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/gemma-4-e2b-it")
@strict
class Gemma4AssistantConfig(PreTrainedConfig):

    model_type = "gemma4_assistant"
    sub_configs = {
        "text_config": Gemma4TextConfig,
    }

    text_config: Gemma4TextConfig | dict[str, Any] | None = None

    backbone_hidden_size: int = 1536
    use_ordered_embeddings: bool = False
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = Gemma4TextConfig(**self.text_config)

        if self.text_config is not None and not self.text_config.num_kv_shared_layers:
            self.text_config.num_kv_shared_layers = self.text_config.num_hidden_layers

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["Gemma4AssistantConfig"]
