
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class VisionTextDualEncoderProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
class VisionTextDualEncoderProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)


__all__ = ["VisionTextDualEncoderProcessor"]
