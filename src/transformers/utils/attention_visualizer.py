import io

import httpx
from PIL import Image

from ..masking_utils import create_causal_mask
from ..models.auto.auto_factory import _get_model_class
from ..models.auto.configuration_auto import AutoConfig
from ..models.auto.modeling_auto import MODEL_FOR_PRETRAINING_MAPPING, MODEL_MAPPING
from ..models.auto.processing_auto import PROCESSOR_MAPPING_NAMES, AutoProcessor
from ..models.auto.tokenization_auto import AutoTokenizer
from .import_utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BLACK_SQUARE = "■"
WHITE_SQUARE = "⬚"


def generate_attention_matrix_from_mask(
    words, mask, img_token="<img>", sliding_window=None, token_type_ids=None, image_seq_length=None
):
    pass


class AttentionMaskVisualizer:
    def __init__(self, model_name: str):
        config = AutoConfig.from_pretrained(model_name)
        self.image_token = "<img>"
        if hasattr(config.get_text_config(), "sliding_window"):
            self.sliding_window = getattr(config.get_text_config(), "sliding_window", None)
        try:
            mapped_cls = _get_model_class(config, MODEL_MAPPING)
        except Exception:
            mapped_cls = _get_model_class(config, MODEL_FOR_PRETRAINING_MAPPING)

        if mapped_cls is None:
            raise ValueError(f"Model name {model_name} is not supported for attention visualization")
        self.mapped_cls = mapped_cls

        class _ModelWrapper(mapped_cls, nn.Module):
            def __init__(self, config, model_name):
                nn.Module.__init__(self)
                self.dummy_module = nn.Linear(1, 1)
                self.config = config

        self.model = _ModelWrapper(config, model_name)
        self.model.to(config.dtype)
        self.repo_id = model_name
        self.config = config

    def __call__(self, input_sentence: str, suffix=""):
        self.visualize_attention_mask(input_sentence, suffix=suffix)

    def visualize_attention_mask(self, input_sentence: str, suffix=""):
        pass
