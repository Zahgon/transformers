
from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring


@auto_docstring
class SiglipProcessor(ProcessorMixin):
    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)


__all__ = ["SiglipProcessor"]
