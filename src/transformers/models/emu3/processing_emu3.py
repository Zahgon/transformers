

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_vision_available
from ...utils.import_utils import requires


if is_vision_available():
    from .image_processing_emu3 import Emu3ImageProcessorKwargs, smart_resize


class Emu3TextKwargs(TextKwargs, total=False):

    return_for_image_generation: bool


class Emu3ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Emu3TextKwargs
    images_kwargs: Emu3ImageProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "return_for_image_generation": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "ratio": "1:1",
            "image_area": 518400,
        },
    }


@auto_docstring
@requires(backends=("vision",))
class Emu3Processor(ProcessorMixin):
    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = tokenizer.image_token  # image_token as placeholder to be replaced by vq-vae tokens
        self.image_token_id = tokenizer.image_token_id
        self.image_start_token = tokenizer.boi_token  # "<|image start|>" fixed tokens for start and end of image
        self.image_end_token = tokenizer.eoi_token  # "<|image end|>"
        self.fake_token_around_image = tokenizer.image_wrapper_token  # "<|image token|>"  every image starts with it
        self.eof_token = tokenizer.eof_token  # "<|extra_201|>"
        self.bos_token = tokenizer.bos_token
        self.downsample_ratio = 8
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[Emu3ProcessorKwargs],
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

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        output_kwargs = self._merge_kwargs(
            Emu3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_for_image_generation = output_kwargs["text_kwargs"].pop("return_for_image_generation", False)
        ratio = output_kwargs["images_kwargs"].pop("ratio", None)
        image_area = output_kwargs["images_kwargs"].pop("image_area", None)

        if return_for_image_generation and images is not None:
            raise ValueError("You should not provide `images` when `return_for_image_generation=True`")

        if not return_for_image_generation and text is None and images is None:
            raise ValueError("You must provide either text or images when `return_for_image_generation=False`")

        image_features = {}
        image_start_tokens = f"{self.image_start_token}"
        image_end_tokens = f"{self.eof_token}{self.image_end_token}"

        if not return_for_image_generation and images is not None:
            image_features = self.image_processor(images, **output_kwargs["images_kwargs"])
            image_sizes = iter(image_features.image_sizes)

            prompt_strings = []
            for sample in text:
                while self.image_token in sample:
                    image_size = next(image_sizes)
                    height, width = image_size
                    height = height // self.downsample_ratio
                    width = width // self.downsample_ratio
                    image_seq_length = height * (width + 1)  # +1 for extra row when converting to BPE in modeling code

                    image_placeholder = f"{image_start_tokens}{height}*{width}{self.fake_token_around_image}{'<placeholder>' * image_seq_length}{image_end_tokens}"
                    sample = sample.replace(self.image_token, image_placeholder, 1)
                    sample = f"{self.bos_token}{sample}"  # add BOS because GPT tokenizer doesn't add it
                prompt_strings.append(sample)
            text = [sample.replace("<placeholder>", self.image_token) for sample in prompt_strings]

        elif return_for_image_generation:
            height, width = self.calculate_generate_size(ratio, image_area, self.downsample_ratio)
            image_prompt = f"{image_start_tokens}{height}*{width}{self.fake_token_around_image}"
            text = [f"{self.bos_token}{sample}{image_prompt}" for sample in text]
            image_features["image_sizes"] = [[height, width]] * len(text)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**text_inputs, **image_features}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass

    def calculate_generate_size(self, ratio, image_area, spatial_factor):
        pass

    def postprocess(self, images: ImageInput, **kwargs):
        return self.image_processor.postprocess(images, **kwargs)

    def post_process_multimodal_output(
        self, generated_outputs, skip_special_tokens=True, generation_mode=None, **kwargs
    ):
        """
        Post-process the output of a multimodal model to return the requested modality output.
        If the model cannot generated the requested modality, an error will be raised.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            generation_mode (`str`, *optional*):
                Generation mode indicated which modality to output and can be one of `["text", "image", "audio"]`.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[Union[str, PIL.Image.Image]]`: The decoded text or generated image.
        """
        if generation_mode is None or generation_mode == "text":
            return self.post_process_image_text_to_text(
                generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs
            )

        elif generation_mode == "image":
            images = self.postprocess(generated_outputs, return_tensors="PIL.Image.Image")
            return images["pixel_values"]

        else:
            raise ValueError(
                f"{self.__class__.__name__} got an unexpected generation_mode={generation_mode}. Supported options are only `text` and `image"
            )


__all__ = ["Emu3Processor"]
