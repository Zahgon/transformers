import re

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging, to_py_obj


logger = logging.get_logger(__name__)


class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "images_kwargs": {
            "do_convert_rgb": True,
            "do_pan_and_scan": False,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
        },
    }


@auto_docstring
class Gemma3Processor(ProcessorMixin):
    valid_processor_kwargs = Gemma3ProcessorKwargs

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        image_seq_length: int = 256,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = f"\n\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\n\n"

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            batched_images = make_nested_list_of_images(images)
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

            if not text:
                text = [" ".join([self.boi_token] * len(images)) for images in batched_images]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            num_crops = to_py_obj(image_inputs.pop("num_crops"))
            batch_num_crops = [[num_crops.pop(0) for _ in range(len(images))] for images in batched_images]
            for batch_idx, (prompt, images, num_crops) in enumerate(zip(text, batched_images, batch_num_crops)):
                image_indexes = [m.start() for m in re.finditer(self.boi_token, prompt)]

                if len(images) != len(image_indexes):
                    raise ValueError(
                        f"Prompt contained {len(image_indexes)} image tokens but received {len(images)} images."
                    )

                for num, idx in reversed(list(zip(num_crops, image_indexes))):
                    if num:
                        formatted_image_text = (
                            f"Here is the original image {self.boi_token} and here are some crops to help you see better "
                            + " ".join([self.boi_token] * num)
                        )
                        prompt = prompt[:idx] + formatted_image_text + prompt[idx + len(self.boi_token) :]
                        text[batch_idx] = prompt

            text = [prompt.replace(self.boi_token, self.full_image_sequence) for prompt in text]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            text_inputs["token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass

    @property
    def model_input_names(self) -> list[str]:
        pass

    @property
    def unused_input_names(self) -> list[str]:
        pass

    @property
    def image_token(self) -> list[str]:
        pass


__all__ = ["Gemma3Processor"]
