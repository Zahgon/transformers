
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class AlignProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "max_length": 64,
        },
    }


@auto_docstring
class AlignProcessor(ProcessorMixin):
    valid_processor_kwargs = AlignProcessorKwargs

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)


__all__ = ["AlignProcessor"]
