
import numpy as np
import torch

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import ChannelDimension, group_images_by_shape, reorder_images, to_pil_image
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_pytesseract_available,
    logging,
    requires_backends,
)


if is_pytesseract_available():
    import pytesseract

from torchvision.transforms.v2 import functional as tvF


logger = logging.get_logger(__name__)


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def apply_tesseract(
    image: "np.ndarray | torch.Tensor",
    lang: str | None,
    tesseract_config: str | None = None,
    input_data_format: str | ChannelDimension | None = None,
):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""
    requires_backends(apply_tesseract, ["pytesseract"])

    if hasattr(image, "cpu"):
        image = image.cpu().numpy()
    elif not isinstance(image, np.ndarray):
        image = np.array(image)

    tesseract_config = tesseract_config if tesseract_config is not None else ""

    pil_image = to_pil_image(image, input_data_format=input_data_format)
    image_width, image_height = pil_image.size
    data = pytesseract.image_to_data(pil_image, lang=lang, output_type="dict", config=tesseract_config)
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

    actual_boxes = []
    for x, y, w, h in zip(left, top, width, height):
        actual_box = [x, y, x + w, y + h]
        actual_boxes.append(actual_box)

    normalized_boxes = []
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"

    return words, normalized_boxes


class LayoutLMv3ImageProcessorKwargs(ImagesKwargs, total=False):

    apply_ocr: bool
    ocr_lang: str | None
    tesseract_config: str | None


@auto_docstring
class LayoutLMv3ImageProcessor(TorchvisionBackend):
    valid_kwargs = LayoutLMv3ImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True
    apply_ocr = True
    ocr_lang = None
    tesseract_config = ""

    def __init__(self, **kwargs: Unpack[LayoutLMv3ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[LayoutLMv3ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        apply_ocr: bool = True,
        ocr_lang: str | None = None,
        tesseract_config: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        if apply_ocr:
            requires_backends(self, "pytesseract")
            words_batch = []
            boxes_batch = []
            for image in images:
                if image.is_cuda:
                    logger.warning_once(
                        "apply_ocr can only be performed on cpu. Tensors will be transferred to cpu before processing."
                    )
                words, boxes = apply_tesseract(
                    image.cpu(), ocr_lang, tesseract_config, input_data_format=ChannelDimension.FIRST
                )
                words_batch.append(words)
                boxes_batch.append(boxes)

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, resample=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        data = BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

        if apply_ocr:
            data["words"] = words_batch
            data["boxes"] = boxes_batch

        return data


__all__ = ["LayoutLMv3ImageProcessor"]
