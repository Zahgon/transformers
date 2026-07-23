from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig


@auto_docstring(checkpoint="google/gemma-4-12b-it")
@strict
class Gemma4UnifiedAssistantConfig(PreTrainedConfig):

    model_type = "gemma4_unified_assistant"
    sub_configs = {
        "text_config": Gemma4UnifiedTextConfig,
    }

    text_config: Gemma4UnifiedTextConfig | dict[str, Any] | None = None

    backbone_hidden_size: int = 3840
    use_ordered_embeddings: bool = False
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)

        if self.text_config is not None and not self.text_config.num_kv_shared_layers:
            self.text_config.num_kv_shared_layers = self.text_config.num_hidden_layers

        super().__post_init__(**kwargs)


__all__ = ["Gemma4UnifiedAssistantConfig"]
