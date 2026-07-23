
import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_transforms import get_resize_output_image_size
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring


@auto_docstring
class LevitImageProcessor(TorchvisionBackend):

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize: shortest_edge is rescaled to int((256/224) * shortest_edge)."""
        if size.shortest_edge:
            shortest_edge = int((256 / 224) * size.shortest_edge)
            new_size_height, new_size_width = get_resize_output_image_size(
                image, size=shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
            size = SizeDict(height=new_size_height, width=new_size_width)
        elif not size.height or not size.width:
            raise ValueError(
                f"Size dict must have keys 'height' and 'width' or 'shortest_edge'. Got {list(size.keys())}."
            )
        return super().resize(image, size=size, resample=resample, **kwargs)


__all__ = ["LevitImageProcessor"]
