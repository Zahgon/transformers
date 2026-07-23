

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_vision_available, logging


if is_vision_available():
    from ...image_utils import load_images

logger = logging.get_logger(__name__)


class GotOcr2TextKwargs(TextKwargs, total=False):

    format: bool | None


class GotOcr2ImagesKwargs(ImagesKwargs, total=False):

    crop_to_patches: bool
    min_patches: int
    max_patches: int
    box: list | tuple[float, float] | tuple[float, float, float, float] | None
    color: str | None
    num_image_tokens: int
    multi_page: bool


class GotOcr2ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: GotOcr2TextKwargs
    images_kwargs: GotOcr2ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "format": False,
        },
        "images_kwargs": {
            "num_image_tokens": 256,
            "multi_page": False,
            "crop_to_patches": False,
            "min_patches": 1,
            "max_patches": 12,
        },
    }


def preprocess_box_annotation(box: list | tuple, image_size: tuple[int, int]) -> list:
    pass


@auto_docstring
class GotOcr2Processor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.message_start_token = "<|im_start|>"
        self.message_end_token = "<|im_end|>"
        self.img_start_token = "<img>"
        self.img_end_token = "</img>"
        self.img_pad_token = "<imgpad>"
        self.image_token = "<imgpad>"  # keep the above for BC, but we need to call it `image_token`
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.system_query = "system\nYou should follow the instructions carefully and explain your answers in detail."

    def _make_list_of_inputs(self, images, text, box, color, multi_page):
        pass

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[GotOcr2ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            GotOcr2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        format_output = output_kwargs["text_kwargs"].pop("format")
        num_image_tokens = output_kwargs["images_kwargs"].pop("num_image_tokens")
        box = output_kwargs["images_kwargs"].pop("box", [None])
        color = output_kwargs["images_kwargs"].pop("color", None)
        multi_page = output_kwargs["images_kwargs"].pop("multi_page")

        crop_to_patches = output_kwargs["images_kwargs"].get("crop_to_patches")
        images, text, box, color = self._make_list_of_inputs(images, text, box, color, multi_page)
        if multi_page:
            num_pages_per_batch = [len(image_group) for image_group in images]
            images = [image for image_group in images for image in image_group]
        else:
            num_pages_per_batch = [1 for _ in range(len(images))]
        images = load_images(images)
        image_sizes = [image.size for image in images]
        image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
        num_patches_array = image_inputs.pop("num_patches")
        if text is None:
            text = []
            patch_indices = np.cumsum(num_pages_per_batch)
            for index, (num_pages, box_single, color_single) in enumerate(zip(num_pages_per_batch, box, color)):
                current_patch_index = patch_indices[index - 1] if index > 0 else 0
                num_patches = sum(num_patches_array[current_patch_index : current_patch_index + num_pages])
                if box_single[0] is not None:
                    box_single = preprocess_box_annotation(box_single, image_sizes[index])
                query = (
                    f"{f'[{color_single}] ' if color_single is not None else ''}"
                    f"{str(box_single) if box_single[0] is not None else ''} "
                    "OCR"
                    f"{' with format' if format_output else ''}"
                    f"{' across multi pages' if multi_page else ''}"
                    f"{' upon the patch reference' if crop_to_patches else ''}"
                    ": "
                )
                prompt = (
                    self.message_start_token
                    + self.system_query
                    + self.message_end_token
                    + self.message_start_token
                    + "user\n"
                    + self.img_start_token
                    + self.img_pad_token * num_image_tokens * num_patches
                    + self.img_end_token
                    + "\n"
                    + query
                    + self.message_end_token
                    + self.message_start_token
                    + "assistant\n"
                )
                text.append(prompt)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)


__all__ = ["GotOcr2Processor"]
