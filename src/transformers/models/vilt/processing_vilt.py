
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class ViltProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_length": False,
            "verbose": True,
        },
    }


@auto_docstring
class ViltProcessor(ProcessorMixin):
    valid_processor_kwargs = ViltProcessorKwargs

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)


__all__ = ["ViltProcessor"]
