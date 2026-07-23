
from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring


@auto_docstring
class XCLIPProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)
        self.video_processor = self.image_processor

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        if videos is not None and images is None:
            images = videos
        return super().__call__(images=images, text=text, **kwargs)


__all__ = ["XCLIPProcessor"]
