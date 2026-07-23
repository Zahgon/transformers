

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class Cohere2VisionProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
            "padding": True,
            "return_mm_token_type_ids": False,
        },
    }


@auto_docstring
class Cohere2VisionProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.patch_size = self.image_processor.patch_size
        self.boi_token = tokenizer.boi_token
        self.eoi_token = tokenizer.eoi_token
        self.image_token = tokenizer.image_token
        self.img_line_break_token = tokenizer.img_line_break_token
        self.image_token_id = tokenizer.image_token_id

    @property
    def image_token_ids(self) -> list[int]:
        pass

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[Cohere2VisionProcessorKwargs],
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
        if text is None:
            raise ValueError("You have to specify text.")
        elif not isinstance(text, (list, tuple)):
            text = [text]

        output_kwargs = self._merge_kwargs(
            Cohere2VisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            batch_num_patches = iter(image_inputs.pop("num_patches"))
            processed_text = []
            for sample in text:
                while self.image_token in sample:
                    num_patches = next(batch_num_patches)
                    img_patches_per_tile = int(self.patch_size**2)

                    img_string = f"{self.boi_token}"
                    for idx in range(1, num_patches):
                        img_string += "<placeholder>" * img_patches_per_tile + self.img_line_break_token
                    img_string += "<placeholder>" * img_patches_per_tile + self.img_line_break_token
                    img_string += f"{self.eoi_token}"

                    sample = sample.replace(self.image_token, img_string, 1)
                processed_text.append(sample)
            text = [sample.replace("<placeholder>", self.image_token) for sample in processed_text]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        pass


__all__ = ["Cohere2VisionProcessor"]
