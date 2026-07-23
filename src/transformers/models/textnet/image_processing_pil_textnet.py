
import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import get_resize_output_image_size
from ...image_transforms import resize as np_resize
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


class TextNetImageProcessorKwargs(ImagesKwargs, total=False):

    size_divisor: int


@auto_docstring
class TextNetImageProcessorPil(PilBackend):

    valid_kwargs = TextNetImageProcessorKwargs

    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 640}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    size_divisor = 32

    def __init__(self, **kwargs: Unpack[TextNetImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[TextNetImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | None",
        size_divisor: int = 32,
        **kwargs,
    ) -> np.ndarray:
        """Resize to shortest_edge then round up to be divisible by size_divisor."""
        if not size.shortest_edge:
            raise ValueError(f"Size must contain 'shortest_edge' key. Got {size.keys()}")
        height, width = get_resize_output_image_size(
            image,
            size=size.shortest_edge,
            default_to_square=False,
            input_data_format=ChannelDimension.FIRST,
        )
        if height % size_divisor != 0:
            height += size_divisor - (height % size_divisor)
        if width % size_divisor != 0:
            width += size_divisor - (width % size_divisor)
        return np_resize(
            image,
            size=(height, width),
            resample=resample,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        return_tensors: str | TensorType | None,
        size_divisor: int = 32,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for TextNet."""
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample, size_divisor=size_divisor)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["TextNetImageProcessorPil"]
