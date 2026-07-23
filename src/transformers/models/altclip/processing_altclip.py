
from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring


@auto_docstring
class AltCLIPProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)


__all__ = ["AltCLIPProcessor"]
