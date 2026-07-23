

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ModernVBERT/colmodernvbert-merged")
@strict
class ColModernVBertConfig(PreTrainedConfig):

    model_type = "colmodernvbert"
    sub_configs = {"vlm_config": PreTrainedConfig}

    vlm_config: dict | PreTrainedConfig | None = None
    embedding_dim: int = 128
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.vlm_config is None:
            self.vlm_config = CONFIG_MAPPING["modernvbert"]()
            logger.info(
                "`vlm_config` is `None`. Initializing `vlm_config` with the `ModernVBertConfig` with default values."
            )
        elif isinstance(self.vlm_config, dict):
            self.vlm_config = CONFIG_MAPPING[self.vlm_config["model_type"]](**self.vlm_config)

        if not hasattr(self.vlm_config, "vocab_size"):
            self.vlm_config.vocab_size = self.vlm_config.get_text_config().vocab_size
        if self.vlm_config is None:
            self.vlm_config = CONFIG_MAPPING["qwen2_vl"]()
            logger.info(
                "`vlm_config` is `None`. Initializing `vlm_config` with the `Qwen2VLConfig` with default values."
            )
        elif isinstance(self.vlm_config, dict):
            self.vlm_config = CONFIG_MAPPING[self.vlm_config["model_type"]](**self.vlm_config)

        if not hasattr(self.vlm_config, "vocab_size"):
            self.vlm_config.vocab_size = self.vlm_config.get_text_config().vocab_size

        super().__post_init__(**kwargs)

    def get_text_config(self, *args, **kwargs) -> PreTrainedConfig:
        return self.vlm_config.get_text_config(*args, **kwargs)


__all__ = ["ColModernVBertConfig"]
