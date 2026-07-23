
from collections.abc import Callable

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    ImageInput,
    PILImageResampling,
    make_flat_list_of_images,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torch_available


IDEFICS_STANDARD_MEAN = [0.48145466, 0.4578275, 0.40821073]
IDEFICS_STANDARD_STD = [0.26862954, 0.26130258, 0.27577711]


class IdeficsImageProcessorKwargs(ImagesKwargs, total=False):

    transform: Callable | None
    image_size: int
    image_num_channels: int


@auto_docstring
class IdeficsImageProcessor(TorchvisionBackend):
    valid_kwargs = IdeficsImageProcessorKwargs
    resample = PILImageResampling.BICUBIC
    image_mean = IDEFICS_STANDARD_MEAN
    image_std = IDEFICS_STANDARD_STD
    size = {"height": 224, "width": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    image_num_channels = 3

    def __init__(self, **kwargs: Unpack[IdeficsImageProcessorKwargs]):
        image_size = kwargs.pop("image_size", None)
        if image_size is not None:
            kwargs["size"] = {"height": image_size, "width": image_size}
        super().__init__(**kwargs)
        self.image_size = self.size.height

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[IdeficsImageProcessorKwargs],
    ) -> "TensorType | BatchFeature":
        transform = kwargs.pop("transform", None)
        if transform is not None:
            if not is_torch_available():
                raise ImportError("To pass in `transform` torch must be installed")
            import torch

            images = self.fetch_images(images)
            images = make_flat_list_of_images(images)
            images = [transform(x) for x in images]
            return torch.stack(images)
        return super().preprocess(images, **kwargs).pixel_values


__all__ = ["IdeficsImageProcessor"]
