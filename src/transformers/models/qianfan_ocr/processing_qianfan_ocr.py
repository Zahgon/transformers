

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, concatenate_list, make_flat_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class QianfanOCRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "crop_to_patches": True,
        },
        "videos_kwargs": {
            "return_tensors": "pt",
        },
    }


@auto_docstring
class QianfanOCRProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        image_seq_length: int = 256,
        chat_template=None,
        image_placeholder_token: str = "<image>",
        **kwargs,
    ):
        r"""
        image_placeholder_token (`str`, *optional*, defaults to `"<image>"`):
            The token emitted by the chat template to mark image positions.
            It is replaced by the full ``<img><IMG_CONTEXT>...<IMG_CONTEXT></img>``
            sequence during processing.
        """
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.image_seq_length = image_seq_length
        self.start_image_token = tokenizer.start_image_token
        self.end_image_token = tokenizer.end_image_token
        self.start_image_token_id = tokenizer.start_image_token_id
        self.end_image_token_id = tokenizer.end_image_token_id
        self.image_token = tokenizer.context_image_token
        self.image_token_id = tokenizer.context_image_token_id
        self.image_ids = [self.image_token_id, self.start_image_token_id, self.end_image_token_id]
        self.image_placeholder_token = image_placeholder_token
        self.video_token = None
        self.video_processor = None

    @property
    def image_token_ids(self) -> list[int]:
        pass

    def _insert_media_placeholders(
        self,
        text: list[str],
        image_pixel_values,
        video_pixel_values,
        image_num_patches: list[int],
        video_num_patches: list[int],
        image_num_patches_indices: np.ndarray,
        video_num_patches_indices: np.ndarray,
        video_patch_indices: np.ndarray,
    ):
        pass

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos=None,
        **kwargs,
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
        if videos is not None:
            raise ValueError("QianfanOCR does not support video input.")
        if text is None:
            raise ValueError("You have to specify text.")

        output_kwargs = self._merge_kwargs(
            QianfanOCRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, (list, tuple)):
            text = [text]

        image_num_patches = []
        image_pixel_values = None
        image_num_patches_indices = np.array([0])
        if images is not None:
            images = self.image_processor.fetch_images(images)
            images = make_flat_list_of_images(images)
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_num_patches = image_inputs.pop("num_patches")
            image_pixel_values = image_inputs.pop("pixel_values")
            image_num_patches_indices = np.cumsum(image_num_patches)

        video_num_patches = []  # per frame
        video_pixel_values = None
        video_patch_indices = np.array([0])
        video_num_patches_indices = np.array([0])
        if videos is not None:
            video_kwargs = output_kwargs["videos_kwargs"]
            video_inputs = self.video_processor(videos=videos, **video_kwargs)
            video_pixel_values = video_inputs.pop("pixel_values_videos")

            batch_size, num_frames, *_ = video_pixel_values.shape
            num_frames_per_video = np.full(batch_size, num_frames)
            num_frames = sum(num_frames_per_video)  # total
            video_patch_indices = np.empty(batch_size + 1, int)
            video_patch_indices[0] = 0
            video_patch_indices[1:] = np.cumsum(num_frames_per_video)
            video_num_patches = [1] * num_frames
            video_num_patches_indices = np.empty(num_frames + 1, int)
            video_num_patches_indices[0] = 0
            video_num_patches_indices[1:] = np.cumsum(video_num_patches)
            video_pixel_values = video_pixel_values.flatten(0, 1)

        image_videos_inputs = {}
        if images is not None or videos is not None:
            text, image_video_patches, image_index, video_index = self._insert_media_placeholders(
                text,
                image_pixel_values,
                video_pixel_values,
                image_num_patches,
                video_num_patches,
                image_num_patches_indices,
                video_num_patches_indices,
                video_patch_indices,
            )
            if images is not None and image_index != len(images):
                raise ValueError("Number of image placeholders in the prompt does not match the number of images.")
            if videos is not None and video_index != len(num_frames_per_video):
                raise ValueError("Number of video placeholders in the prompt does not match the number of videos.")

            image_videos_inputs = {"pixel_values": concatenate_list(image_video_patches)}

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**text_inputs, **image_videos_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass

    @property
    def model_input_names(self):
        pass


__all__ = ["QianfanOCRProcessor"]
