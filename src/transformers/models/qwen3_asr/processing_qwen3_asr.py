
import unicodedata

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio_chat_template
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, prepare_prompt_input
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring
from ...utils.import_utils import is_nagisa_available, is_soynlp_available


LANGUAGE_CODE_TO_NAME = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}

FORCED_ALIGNER_LANGUAGES = {
    "Chinese", "English", "Cantonese", "French", "German",
    "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
}

SUPPORTED_LANGUAGE_NAMES = set(LANGUAGE_CODE_TO_NAME.values())


def resolve_language(language: str | None) -> str | None:
    pass


def _prepare_language_inputs(
    language: str | list[str] | None, batch_size: int, allow_broadcast: bool = False
) -> list[str | None]:
    pass


def _audio_content_item(audio_item) -> dict:
    pass


def _is_cjk_char(char: str) -> bool:
    pass


def _is_kept_char(char: str) -> bool:
    pass


def _clean_tokens(raw_tokens) -> list[str]:
    pass


def _parse_single_output(text: str) -> dict:
    """Parse a single decoded ASR string into language + transcription like the original implementation."""
    if text is None or not str(text).strip():
        return {"language": None, "transcription": ""}
    text = str(text).strip()

    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[-1]

    text = _detect_and_fix_repetitions(text)

    marker = "<asr_text>"
    if marker not in text:
        return {"language": None, "transcription": text.strip()}

    prefix, transcription = text.split(marker, 1)
    prefix = prefix.strip()

    if prefix.lower() == "language none":
        return {"language": None, "transcription": transcription.strip()}

    language = None
    for line in prefix.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("language "):
            val = line[len("language ") :].strip()
            if val:
                language = val
        else:
            language = line
        break  # only inspect the first non-empty line, matching the original

    return {"language": language or None, "transcription": transcription.strip()}


def _fix_timestamps(raw: np.ndarray) -> list[int]:
    pass


def _detect_and_fix_repetitions(text, threshold=20):
    """
    Original implementation uses this post-processing to remove repeated characters and patterns in the ASR output
    https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/utils.py#L432
    """

    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1

            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i : i + count])
                i += count
        return "".join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s

        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break

                pattern = s[i : i + k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx : start_idx + k] != pattern:
                        valid = False
                        break

                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index : end_index + k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break

            if found:
                break
            else:
                result.append(s[i])
                i += 1

        if not found:
            result.append(s[i:])
        return "".join(result)

    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


class Qwen3ASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "return_attention_mask": True,
            "n_window": 50,  # should match config.n_window
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class Qwen3ASRProcessor(ProcessorMixin):
    valid_processor_kwargs = Qwen3ASRProcessorKwargs

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        timestamp_segment_time: float = 80,
    ):
        r"""
        timestamp_segment_time (`float`, *optional*):
            Milliseconds per timestamp class. Defaults to 80 ms.
        """
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        self.timestamp_segment_time = timestamp_segment_time
        self.audio_token = self.tokenizer.audio_token
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        self.audio_eos_token = self.tokenizer.audio_eos_token
        self.audio_eos_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_eos_token)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput,
        output_labels: bool | None = False,
        **kwargs: Unpack[Qwen3ASRProcessorKwargs],
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

    def _get_audio_token_length(self, audio_lengths, n_window=50):
        pass

    def _process_audio(self, audio: AudioInput, **kwargs):
        pass

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        pass

    def apply_transcription_request(
        self,
        audio: AudioInput | list[AudioInput],
        language: str | list[str] | None = None,
        prompt: str | list[str] | None = None,
        **kwargs: Unpack[Qwen3ASRProcessorKwargs],
    ) -> BatchFeature:
        pass

    def decode(self, *args, return_format="raw", **kwargs):
        """
        Forward arguments to the tokenizer's decode and optionally parse the ASR output.

        Qwen3 ASR outputs transcription in the format: ``language <LANG><asr_text>transcribed text``

        Args:
            return_format (`str`, *optional*, defaults to `"raw"`):
                Options:

                - ``"raw"``: Return raw decoded strings from the tokenizer.
                - ``"parsed"``: Return a dict (or list of dicts) with ``"language"`` and ``"transcription"`` keys.
                - ``"transcription_only"``: Extract only the transcribed text (after ``<asr_text>``).

                ``skip_special_tokens`` is hard-set to ``True`` for ``"parsed"`` and ``"transcription_only"``.
        """
        valid_formats = ["raw", "parsed", "transcription_only"]
        if return_format not in valid_formats:
            raise ValueError(f"return_format must be one of {valid_formats}.")
        if return_format != "raw":
            kwargs["skip_special_tokens"] = True

        decoded = self.tokenizer.decode(*args, **kwargs)
        if return_format == "parsed":
            decoded = self.parse_output(decoded)
        elif return_format == "transcription_only":
            decoded = self.extract_transcription(decoded)
        return decoded

    def parse_output(self, text: str | list[str]) -> dict | list[dict]:
        """
        Parse Qwen3 ASR raw output into a structured dict.

        The model outputs ``language <LANG><asr_text>transcribed text``.
        This method returns a dict with ``"language"`` and ``"transcription"`` keys.

        Args:
            text (`str` or `list[str]`): Raw decoded output(s).

        Returns:
            `dict` or `list[dict]`: Parsed output(s). Each dict has keys
            ``"language"`` (str or None) and ``"transcription"`` (str).
            Returns the original string as the transcription if parsing fails.
        """
        if isinstance(text, str):
            return _parse_single_output(text)
        return [_parse_single_output(raw_text) for raw_text in text]

    def extract_transcription(self, text: str | list[str]) -> str | list[str]:
        """
        Extract transcription text from Qwen3 ASR raw output.

        The model outputs ``language <LANG><asr_text>transcribed text``.
        This method extracts the text after ``<asr_text>``.

        Args:
            text (`str` or `list[str]`): Raw decoded output(s).

        Returns:
            `str` or `list[str]`: Extracted transcription(s). Returns the
            original string if ``<asr_text>`` is not found.
        """
        if isinstance(text, str):
            return _parse_single_output(text)["transcription"]
        return [_parse_single_output(raw_text)["transcription"] for raw_text in text]

    def split_words_for_alignment(self, text: str | list[str], language: str | None = None) -> list[str]:
        pass

    def prepare_forced_aligner_inputs(
        self,
        audio: AudioInput,
        transcript: str | list[str],
        language: str | list[str] | None = None,
        **kwargs,
    ) -> tuple[BatchFeature, list[list[str]]]:
        pass

    def decode_forced_alignment(
        self,
        logits,
        input_ids,
        word_lists: list[list[str]],
        timestamp_token_id: int,
        timestamp_segment_time: float | None = None,
    ) -> list[list[dict]]:
        pass

    @property
    def unused_input_names(self) -> list[str]:
        pass

    @property
    def model_input_names(self):
        pass

    @property
    def audio_token_ids(self):
        pass


__all__ = ["Qwen3ASRProcessor"]
