
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class TvpProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "truncation": True,
            "padding": "max_length",
            "pad_to_max_length": True,
            "return_token_type_ids": False,
        },
    }


@auto_docstring
class TvpProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)
        self.video_processor = image_processor

    def post_process_video_grounding(self, logits, video_durations):
        pass


__all__ = ["TvpProcessor"]
