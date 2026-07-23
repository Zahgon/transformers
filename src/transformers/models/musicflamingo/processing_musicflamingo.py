
import numpy as np

from ...audio_utils import AudioInput
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, is_torch_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class MusicFlamingoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "return_attention_mask": True,
            "padding": "max_length",
        },
        "common_kwargs": {
            "return_tensors": "pt",
            "padding_side": "left",
        },
    }


@requires(backends=("torch",))
@auto_docstring
class MusicFlamingoProcessor(ProcessorMixin):
    valid_processor_kwargs = MusicFlamingoProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<sound>",
        audio_bos_token="<|sound_bos|>",
        audio_eos_token="<|sound_eos|>",
        max_audio_len=1200,
    ):
        r"""
        audio_token (`Optional[str]`, *optional*, defaults to `"<sound>"`):
            Special token used to represent audio inputs in the chat template.
        audio_bos_token (`Optional[str]`, *optional*, defaults to `"<|sound_bos|>"`):
            Special token used to represent the beginning of audio.
        audio_eos_token (`Optional[str]`, *optional*, defaults to `"<|sound_eos|>"`):
            Special token used to represent the end of audio.
        max_audio_len (`int`, *optional*, defaults to 1200):
            Maximum length of audio sequences in seconds. Audio longer than this will be truncated.
        """
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        self.max_audio_len = max_audio_len
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.audio_bos_token_id = tokenizer.convert_tokens_to_ids(audio_bos_token)
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids(audio_eos_token)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[MusicFlamingoProcessorKwargs],
    ) -> BatchFeature:
        r"""
        output_labels (bool, *optional*, default=False):
            Whether to return labels for training.

        Returns:
            [`BatchFeature`]: A dictionary with tokenized text (`input_ids`, `attention_mask`) and
            audio features (`input_features`, `input_features_mask`).
        """
        if "return_tensors" in kwargs and kwargs["return_tensors"] != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if output_labels:
            kwargs["return_mm_token_type_ids"] = True
        model_inputs = super().__call__(audio=audio, text=text, **kwargs)

        if output_labels:
            mm_token_type_ids = model_inputs.pop("mm_token_type_ids")
            labels = model_inputs["input_ids"].clone()
            labels[mm_token_type_ids != 0] = -100  # audio positions
            labels[labels == self.tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels
        return BatchFeature(data=model_inputs, tensor_type="pt")

    def validate_inputs(
        self,
        audio: AudioInput | None = None,
        text: TextInput | list[TextInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        pass

    def _get_audio_token_length(self, audio_lengths):
        pass

    def _process_audio(self, audio: AudioInput, **kwargs):
        pass

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        pass

    @property
    def model_input_names(self) -> list[str]:
        pass

    @property
    def unused_input_names(self) -> list[str]:
        pass

    @property
    def audio_token_ids(self):
        pass

    @property
    def audio_ids(self):
        pass


__all__ = ["MusicFlamingoProcessor"]
