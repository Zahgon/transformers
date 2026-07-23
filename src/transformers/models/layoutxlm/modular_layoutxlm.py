

from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ..layoutlmv2.configuration_layoutlmv2 import LayoutLMv2Config


@auto_docstring(checkpoint="microsoft/layoutxlm-base")
@strict
class LayoutXLMConfig(LayoutLMv2Config):

    pass


__all__ = ["LayoutXLMConfig"]
