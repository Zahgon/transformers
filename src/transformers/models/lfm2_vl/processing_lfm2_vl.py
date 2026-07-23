import math

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from ...tokenization_utils_base import BatchEncoding, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class Lfm2VlTextKwargs(TextKwargs, total=False):

    use_image_special_tokens: bool | None


class Lfm2VlProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Lfm2VlTextKwargs
    _defaults = {
        "images_kwargs": {
            "return_row_col_info": True,
        },
        "text_kwargs": {
            "use_image_special_tokens": True,
            "add_special_tokens": False,
            "padding": False,
            "is_split_into_words": False,
        },
    }


@auto_docstring
class Lfm2VlProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template: str | None = None,
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = (
            tokenizer.image_token_id
            if hasattr(tokenizer, "image_token_id")
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.image_start_token = getattr(tokenizer, "image_start_token", "<|image_start|>")
        self.image_end_token = getattr(tokenizer, "image_end_token", "<|image_end|>")
        self.image_thumbnail_token = getattr(tokenizer, "image_thumbnail_token", "<|img_thumbnail|>")
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] | None = None,
        text: TextInput | list[TextInput] | None = None,
        **kwargs: Unpack[Lfm2VlProcessorKwargs],
    ) -> BatchEncoding:
        if text is None and images is None:
            raise ValueError("You must provide one of `text` or `images`.")

        if images is not None and text is None:
            raise ValueError(
                "You must provide `text` when `images` is provided. Minimal text consists of a single image token."
            )

        output_kwargs = self._merge_kwargs(
            Lfm2VlProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        n_images_in_text = [sample.count(self.image_token) for sample in text]
        if sum(n_images_in_text) > 0 and images is None:
            raise ValueError(f"We detected {sum(n_images_in_text)} tokens in the text but no images were passed")

        inputs = {}
        use_image_special_tokens = output_kwargs["text_kwargs"].pop("use_image_special_tokens")

        if images is not None:
            images = self.image_processor.fetch_images(images)
            batched_images = make_nested_list_of_images(images)
            vision_inputs = self.image_processor(batched_images, **output_kwargs["images_kwargs"])

            n_images_in_images = [len(sublist) for sublist in batched_images]
            if n_images_in_images != n_images_in_text:
                raise ValueError(
                    f"The number of images in the text {n_images_in_text} and images {n_images_in_images} should be the same."
                )

            text = self.expand_text_with_placeholders(
                text,
                batched_images,
                image_rows=vision_inputs.pop("image_rows"),
                image_cols=vision_inputs.pop("image_cols"),
                image_sizes=vision_inputs.pop("image_sizes"),
                use_image_special_tokens=use_image_special_tokens,
                **output_kwargs["images_kwargs"],
            )
            inputs.update(vision_inputs)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        inputs.update(text_inputs)

        return BatchFeature(inputs, tensor_type=return_tensors)

    def expand_text_with_placeholders(
        self,
        text: list[str],
        images: list[list[ImageInput]],
        image_rows: list[list[int]],
        image_cols: list[list[int]],
        image_sizes: list[list[int]],
        use_image_special_tokens: bool,
        **images_kwargs,
    ) -> list[str]:
        pass

    def _build_image_tokens(
        self,
        rows: int,
        cols: int,
        tokens_per_tile: int,
        tokens_for_image: int,
        use_thumbnail: bool,
        use_image_special_tokens: bool,
    ) -> str:
        pass

    def _compute_tokens_per_tile(self, tile_size: int, encoder_patch_size: int, downsample_factor: int) -> int:
        pass

    def _compute_tokens_for_image(self, image_size: list[int], encoder_patch_size: int, downsample_factor: int) -> int:
        pass

    def _get_image_num_tokens(self, image_size: list[int], **images_kwargs) -> tuple[int, int]:
        pass

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LFM2Tokeniser's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        batched_decode_output = self.tokenizer.batch_decode(*args, **kwargs)
        return batched_decode_output

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LFM2Tokeniser's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        decode_output = self.tokenizer.decode(*args, **kwargs)
        return decode_output

    @property
    def model_input_names(self):
        pass


__all__ = ["Lfm2VlProcessor"]
