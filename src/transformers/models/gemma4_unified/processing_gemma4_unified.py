import math

import numpy as np

from ...audio_utils import AudioInput
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    auto_docstring,
    is_vision_available,
    logging,
)
from ...utils.import_utils import requires
from ...video_utils import VideoInput


if is_vision_available():
    from .image_processing_gemma4_unified import Gemma4UnifiedImageProcessorKwargs, get_aspect_ratio_preserving_size


logger = logging.get_logger(__name__)


class Gemma4UnifiedVideoProcessorKwargs(VideosKwargs, total=False):

    patch_size: int
    max_soft_tokens: int
    pooling_kernel_size: int


class Gemma4UnifiedProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Gemma4UnifiedImageProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "return_mm_token_type_ids": True,
        },
        "images_kwargs": {
            "do_convert_rgb": True,
        },
        "audio_kwargs": {},
        "videos_kwargs": {"return_metadata": True},
    }


@auto_docstring
@requires(backends=("vision",))
class Gemma4UnifiedProcessor(ProcessorMixin):
    valid_processor_kwargs = Gemma4UnifiedProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        image_processor,
        tokenizer,
        video_processor,
        chat_template=None,
        image_seq_length: int = 280,
        audio_seq_length: int = 750,
        audio_ms_per_token: int = 40,
        **kwargs,
    ):
        r"""
        image_seq_length (`int`, *optional*, defaults to 280):
            The number of soft tokens per image used for placeholder expansion.
        audio_seq_length (`int`, *optional*, defaults to 750):
            The maximum number of audio soft tokens per audio segment. Serves as an
            upper-bound cap when dynamic audio token counts are computed.
        audio_ms_per_token (`int`, *optional*, defaults to 40):
            Milliseconds of audio per output soft token. Used to dynamically compute
            the number of audio placeholder tokens as ``ceil(duration_ms / audio_ms_per_token)``.
            The default of 40 comes from the SSCP convolution's 4× time reduction on 10ms frames.
        """
        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.eoi_token = tokenizer.eoi_token
        self.image_token = tokenizer.image_token

        tokenizer.add_special_tokens({"additional_special_tokens": ["<|video|>"]})
        self.video_token = "<|video|>"
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)

        self.audio_seq_length = audio_seq_length
        self.audio_ms_per_token = audio_ms_per_token
        self.audio_token_id = getattr(tokenizer, "audio_token_id", None)
        self.audio_token = getattr(tokenizer, "audio_token", None)
        self.boa_token = getattr(tokenizer, "boa_token", None)
        self.eoa_token = getattr(tokenizer, "eoa_token", None)

        super().__init__(
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs,
    ):
        pass

    def validate_inputs(
        self,
        images: ImageInput | list[ImageInput] | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        pass

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        pass

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        pass

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        pass

    def _get_num_multimodal_tokens(self, image_sizes=None, audio_lengths=None, **kwargs):
        pass

    def _compute_audio_num_tokens(self, audio_waveform, sampling_rate: int) -> int:
        pass

    @property
    def model_input_names(self):
        pass

    @property
    def unused_input_names(self) -> list[str]:
        pass


__all__ = ["Gemma4UnifiedProcessor"]
