
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    ImageInput,
    ImageType,
    PILImageResampling,
    SizeDict,
    get_image_type,
    is_pil_image,
    is_valid_image,
    to_numpy_array,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires


if TYPE_CHECKING:
    import torch

    from .modeling_superglue import SuperGlueKeypointMatchingOutput


def is_grayscale(image: np.ndarray):
    if image.shape[0] == 1:
        return True
    return np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...])


def convert_to_grayscale(image: ImageInput) -> ImageInput:
    """
    Converts an image to grayscale format using the NTSC formula. Only support numpy and PIL Image.

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (Image):
            The image to convert.
    """

    if isinstance(image, np.ndarray):
        if is_grayscale(image):
            return image

        gray_image = image[0, ...] * 0.2989 + image[1, ...] * 0.5870 + image[2, ...] * 0.1140
        gray_image = np.stack([gray_image] * 3, axis=0)
        return gray_image

    if not isinstance(image, Image.Image):
        return image

    image = image.convert("L")
    return image


def validate_and_format_image_pairs(images: ImageInput):
    error_message = (
        "Input images must be a one of the following :",
        " - A pair of PIL images.",
        " - A pair of 3D arrays.",
        " - A list of pairs of PIL images.",
        " - A list of pairs of 3D arrays.",
    )

    def _is_valid_image(image):
        """images is a PIL Image or a 3D array."""
        return is_pil_image(image) or (
            is_valid_image(image) and get_image_type(image) != ImageType.PIL and len(image.shape) == 3
        )

    if isinstance(images, list):
        if len(images) == 2 and all((_is_valid_image(image)) for image in images):
            return images
        if all(
            isinstance(image_pair, list)
            and len(image_pair) == 2
            and all(_is_valid_image(image) for image in image_pair)
            for image_pair in images
        ):
            return [image for image_pair in images for image in image_pair]
    raise ValueError(error_message)


class SuperGlueImageProcessorKwargs(ImagesKwargs, total=False):

    do_grayscale: bool


@auto_docstring
class SuperGlueImageProcessorPil(PilBackend):
    valid_kwargs = SuperGlueImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    size = {"height": 480, "width": 640}
    default_to_square = False
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = None
    do_grayscale = True

    def __init__(self, **kwargs: Unpack[SuperGlueImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[SuperGlueImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _prepare_images_structure(self, images: ImageInput, **kwargs) -> ImageInput:
        images = self.fetch_images(images)
        return validate_and_format_image_pairs(images)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | None,
        do_rescale: bool,
        rescale_factor: float,
        return_tensors: str | TensorType | None,
        do_grayscale: bool = True,
        **kwargs,
    ) -> BatchFeature:
        all_images = []
        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)

            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor)

            if do_grayscale:
                image = convert_to_grayscale(image)

            all_images.append(image)

        image_pairs = [all_images[i : i + 2] for i in range(0, len(all_images), 2)]

        data = {"pixel_values": image_pairs}

        return BatchFeature(data=data, tensor_type=return_tensors)

    @requires(backends=("torch",))
    def post_process_keypoint_matching(
        self,
        outputs: "SuperGlueKeypointMatchingOutput",
        target_sizes: TensorType | list[tuple],
        threshold: float = 0.0,
    ) -> list[dict[str, "torch.Tensor"]]:
        """
        Converts the raw output of [`SuperGlueKeypointMatchingOutput`] into lists of keypoints, scores and descriptors
        with coordinates absolute to the original image sizes.
        Args:
            outputs ([`SuperGlueKeypointMatchingOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` or `list[tuple[tuple[int, int]]]`, *optional*):
                Tensor of shape `(batch_size, 2, 2)` or list of tuples of tuples (`tuple[int, int]`) containing the
                target size `(height, width)` of each image in the batch. This must be the original image size (before
                any processing).
            threshold (`float`, *optional*, defaults to `0.0`):
                Threshold to filter out the matches with low scores.
        Returns:
            `list[Dict]`: A list of dictionaries, each dictionary containing the keypoints in the first and second image
            of the pair, the matching scores and the matching indices.
        """
        import torch

        if outputs.mask.shape[0] != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the mask")
        if not all(len(target_size) == 2 for target_size in target_sizes):
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        if isinstance(target_sizes, list):
            image_pair_sizes = torch.tensor(target_sizes, device=outputs.mask.device)
        else:
            if target_sizes.shape[1] != 2 or target_sizes.shape[2] != 2:
                raise ValueError(
                    "Each element of target_sizes must contain the size (h, w) of each image of the batch"
                )
            image_pair_sizes = target_sizes

        keypoints = outputs.keypoints.clone()
        keypoints = keypoints * image_pair_sizes.flip(-1).reshape(-1, 2, 1, 2)
        keypoints = keypoints.to(torch.int32)

        results = []
        for mask_pair, keypoints_pair, matches, scores in zip(
            outputs.mask, keypoints, outputs.matches[:, 0], outputs.matching_scores[:, 0]
        ):
            mask0 = mask_pair[0] > 0
            mask1 = mask_pair[1] > 0
            keypoints0 = keypoints_pair[0][mask0]
            keypoints1 = keypoints_pair[1][mask1]
            matches0 = matches[mask0]
            scores0 = scores[mask0]

            valid_matches = (scores0 > threshold) & (matches0 > -1) & (matches0 < keypoints1.shape[0])

            matched_keypoints0 = keypoints0[valid_matches]
            matched_keypoints1 = keypoints1[matches0[valid_matches]]
            matching_scores = scores0[valid_matches]

            results.append(
                {
                    "keypoints0": matched_keypoints0,
                    "keypoints1": matched_keypoints1,
                    "matching_scores": matching_scores,
                }
            )

        return results

    def visualize_keypoint_matching(
        self, images: ImageInput, keypoint_matching_output: list[dict[str, "torch.Tensor"]]
    ) -> list["Image.Image"]:
        pass

    def _get_color(self, score):
        pass


__all__ = ["SuperGlueImageProcessorPil"]
