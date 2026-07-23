
from ...image_processing_backends import PilBackend
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring


@auto_docstring
class CLIPImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        if "use_square_size" in kwargs and kwargs["use_square_size"]:
            kwargs["size"] = {"height": self.size["shortest_edge"], "width": self.size["shortest_edge"]}
            kwargs.pop("use_square_size")

        super().__init__(**kwargs)


__all__ = ["CLIPImageProcessorPil"]
