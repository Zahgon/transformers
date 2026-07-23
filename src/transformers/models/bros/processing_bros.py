
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class BrosProcessorKwargs(ProcessingKwargs, total=False):
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
class BrosProcessor(ProcessorMixin):
    valid_processor_kwargs = BrosProcessorKwargs

    def __init__(self, tokenizer=None, **kwargs):
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(tokenizer)


__all__ = ["BrosProcessor"]
