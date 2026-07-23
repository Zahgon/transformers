

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio_chat_template
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, is_torch_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class GlmAsrProcessorKwargs(ProcessingKwargs, total=False):
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
class GlmAsrProcessor(ProcessorMixin):
    valid_processor_kwargs = GlmAsrProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<|pad|>",
        default_transcription_prompt="Please transcribe this audio into text",
        max_audio_len=655,
    ):
        r"""
        audio_token (`Optional[str]`, *optional*, defaults to `"<|pad|>`"):
            Special token used to represent audio inputs in the chat template.
        default_transcription_prompt (`str`, *optional*, defaults to `"Please transcribe this audio into text"`):
            Default prompt to use for transcription tasks when applying transcription requests.
        max_audio_len (`int`, *optional*, defaults to 655):
            Maximum length of audio sequences in seconds. Audio longer than this will be truncated.
            655 gives approximately 8192 tokens, corresponding to the maximum sequence length of the text model.
        """
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        self.default_transcription_prompt = default_transcription_prompt
        self.max_audio_len = max_audio_len
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[GlmAsrProcessorKwargs],
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

    def _get_audio_token_length(self, audio_lengths: "torch.Tensor") -> "torch.Tensor":
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

    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        prompt: str | list[str] | None = None,
        **kwargs: Unpack[GlmAsrProcessorKwargs],
    ) -> BatchFeature:
        pass

    def decode(self, *args, strip_prefix=False, **kwargs):
        """
        Forward arguments to [`~PreTrainedTokenizer.decode`] and optionally remove the assistant framing the model
        was trained to produce.

        AF3 transcription requests respond with sentences such as `"The spoken content of the audio is \"...\"."`.
        Setting `strip_prefix=True` trims the fixed prefix for just the transcription text.
        """
        decoded = self.tokenizer.decode(*args, **kwargs)
        if strip_prefix:
            decoded = [self._strip_assistant_prefix_and_quotes(text) for text in decoded]
        return decoded

    def batch_decode(self, *args, **kwargs):
        """BC as previous examples used batch_decode"""
        return self.decode(*args, **kwargs)

    def _strip_assistant_prefix_and_quotes(self, text: str) -> str:
        """
        Remove the assistant prefix and surrounding quotes from a decoded transcription string.
        """

        stripped = text.strip()

        for prefix in (
            "The spoken content of the audio is",
            "The transcription of the audio is",
            "The content of the input audio is",
        ):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :].strip()
                break

        if stripped.endswith("."):
            stripped = stripped[:-1].strip()

        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
            stripped = stripped[1:-1].strip()

        return stripped


__all__ = ["GlmAsrProcessor"]
