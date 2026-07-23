
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class Siglip2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "truncation": True,
            "max_length": 64,
        },
        "images_kwargs": {
            "max_num_patches": 256,
            "patch_size": 16,
        },
    }


@auto_docstring
class Siglip2Processor(ProcessorMixin):
    valid_processor_kwargs = Siglip2ProcessorKwargs

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)


__all__ = ["Siglip2Processor"]
