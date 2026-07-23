
from collections.abc import Iterable

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, get_image_size, to_numpy_array
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)


class PerceptionLMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


@auto_docstring
class PerceptionLMProcessor(ProcessorMixin):
    def __init__(
        self,
        video_processor=None,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        chat_template=None,
        pooling_ratio=2,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        pooling_ratio (`int`, *optional*, defaults to 2):
            Pooling ratio for vision tokens. If not 1, 2D adaptive pooling is applied over projected vision tokens.
        """
        self.patch_size = patch_size
        self.pooling_ratio = pooling_ratio
        self.image_token = tokenizer.image_token
        self.video_token = tokenizer.video_token
        self.image_token_id = tokenizer.image_token_id
        self.video_token_id = tokenizer.video_token_id
        super().__init__(video_processor, image_processor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[PerceptionLMProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is provided.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is provided).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is provided.
            - **pixel_values_videos** -- Video pixel values to be fed to a model. Returned when `videos` is provided.
        """
        if text is None:
            raise ValueError(
                "You have to specify at least `text` input. Optionally, you can also specify `images` or `videos`."
            )

        output_kwargs = self._merge_kwargs(
            PerceptionLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if videos is not None:
            videos_inputs = self.video_processor(videos, **output_kwargs["videos_kwargs"])
        else:
            videos_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        prompt_strings = []

        pixel_values = iter(image_inputs.get("pixel_values", []))
        pixel_values_videos = iter(videos_inputs.get("pixel_values_videos", []))
        for sample in text:
            sample = self._expand_media_tokens(sample, self.tokenizer.image_token, pixel_values)
            sample = self._expand_media_tokens(sample, self.tokenizer.video_token, pixel_values_videos)
            prompt_strings.append(sample)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    def _expand_media_tokens(self, sample, media_token: str, media_iter: Iterable):
        pass

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass


__all__ = ["PerceptionLMProcessor"]
