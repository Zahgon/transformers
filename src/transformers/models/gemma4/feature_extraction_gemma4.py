
import math
import warnings
from collections.abc import Sequence

import numpy as np

from ...audio_utils import mel_filter_bank, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


def _unfold(array: np.ndarray, dimension: int, size: int, step: int) -> np.ndarray:
    pass


class Gemma4AudioFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        frame_length_ms: float = 20.0,
        hop_length_ms: float = 10.0,
        min_frequency: float = 0.0,
        max_frequency: float = 8000.0,
        preemphasis: float = 0.0,
        preemphasis_htk_flavor: bool = True,
        fft_overdrive: bool = False,
        dither: float = 0.0,
        input_scale_factor: float = 1.0,
        mel_floor: float = 1e-3,
        per_bin_mean: Sequence[float] | None = None,
        per_bin_stddev: Sequence[float] | None = None,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.preemphasis = preemphasis
        self.preemphasis_htk_flavor = preemphasis_htk_flavor
        self.fft_overdrive = fft_overdrive
        self.dither = dither
        self.input_scale_factor = input_scale_factor
        self.frame_length = int(round(sampling_rate * frame_length_ms / 1000.0))
        self.hop_length = int(round(sampling_rate * hop_length_ms / 1000.0))
        self.mel_floor = np.array(mel_floor, dtype=np.float64)

        fft_length = 2 ** math.ceil(math.log2(self.frame_length))
        if self.fft_overdrive:
            fft_length *= 2
        self.fft_length = fft_length

        self.window = window_function(self.frame_length).astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mel_filters = mel_filter_bank(
                num_frequency_bins=self.fft_length // 2 + 1,
                num_mel_filters=feature_size,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                sampling_rate=self.sampling_rate,
                norm=None,
                mel_scale="htk",
            )

        if per_bin_mean is not None:
            self.per_bin_mean = np.array(per_bin_mean).reshape(1, 1, feature_size)
        else:
            self.per_bin_mean = None

        if per_bin_stddev is not None:
            self.per_bin_stddev = np.array(per_bin_stddev).reshape(1, 1, feature_size)
        else:
            self.per_bin_stddev = None

    def _extract_spectrogram(self, waveform: np.ndarray, attention_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy = "longest",
        max_length: int | None = 480_000,
        truncation: bool = True,
        pad_to_multiple_of: int | None = 128,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = True,
        **kwargs,
    ) -> BatchFeature:
        """Creates a batch of MEL spectrograms from the provided raw speech.

        This implementation uses a different algorithm for windowing and preemphasis compared to the built-in
        `transformers.audio_utils.spectrogram()` function that _will_ result in different outputs. Consider this
        carefully when selecting an audio feature extractor, especially with pre-trained models.

        Args:
            raw_speech:
                The audio for which MEL spectrograms are created.
            padding (`Union[bool, str, PaddingStrategy]`, *optional*, defaults to `"longest"`):
                The padding strategy to use for batches of audio with different lengths.
            max_length (`int`, *optional*, defaults to 480000):
                If provided, defines the maximum length of the audio to allow. Audio longer than this will be
                truncated if `truncation=True`.
            truncation (`bool`, *optional*, defaults to `True`):
                Whether or not to truncate audio above `max_length`.
            pad_to_multiple_of (`int`, *optional*, defaults to 128):
                When padding, pad to a multiple of this value. The default value is defined for optimal TPU support.
            return_tensors (`Union[str, TensorType]`, *optional*, defaults to `None`):
                The type of tensors to return (e.g., NumPy, or Torch).
            return_attention_mask (`bool`, *optional*, defaults to `True`):
                Whether to return the attention mask for the generated MEL spectrograms.
        """

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        is_batched_sequence = isinstance(raw_speech, Sequence) and isinstance(raw_speech[0], (np.ndarray, Sequence))
        is_batched = is_batched_numpy or is_batched_sequence

        if is_batched:
            raw_speech = [np.asarray([rs]).T for rs in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech)

        if not is_batched:  # always return a batch
            raw_speech = [np.asarray([raw_speech])]

        batched_speech = self.pad(
            BatchFeature({"input_features": raw_speech}),
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        prepared_speech = []
        prepared_speech_mask = []
        for speech, mask in zip(batched_speech.input_features, batched_speech.attention_mask):
            speech, mask = self._extract_spectrogram(speech.T, mask)
            prepared_speech.append(speech.astype(np.float32))
            prepared_speech_mask.append(mask)

        prepared_speech = [speech * mask[..., None] for speech, mask in zip(prepared_speech, prepared_speech_mask)]

        return BatchFeature(
            {"input_features": prepared_speech, "input_features_mask": prepared_speech_mask},
            tensor_type=return_tensors,
        )


__all__ = ["Gemma4AudioFeatureExtractor"]
