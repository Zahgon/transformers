
import math
from collections.abc import Sequence

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


def create_fb_matrix(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    fft_length: int,
    norm: str | None = None,
) -> np.ndarray:
    r"""Create a frequency bin conversion matrix (NumPy version).

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        fft_length (int): FFT length
        norm (Optional[str]): If 'slaney', divide the triangular mel weights by
          the width of the mel band (area normalization). (Default: ``None``)

    Returns:
        np.ndarray: Triangular filter banks (fb matrix) of size (``n_freqs``,
        ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of
        filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A @ create_fb_matrix_numpy(A.shape[-1], ...)``.
    """

    if norm is not None and norm != "slaney":
        raise ValueError("norm must be one of None or 'slaney'")

    all_freqs = np.arange(n_freqs, dtype=np.float32) * (sample_rate / fft_length)

    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = np.expand_dims(f_pts, 0) - np.expand_dims(all_freqs, 1)  # (n_freqs, n_mels + 2)
    zero = np.zeros(1, dtype=np.float32)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = np.maximum(zero, np.minimum(down_slopes, up_slopes))

    if norm is not None and norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= np.expand_dims(enorm, 0)

    return fb


def _unfold(array: np.ndarray, dimension: int, size: int, step: int) -> np.ndarray:
    pass


class Gemma3nAudioFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        frame_length_ms: float = 32.0,
        hop_length_ms: float = 10.0,
        min_frequency: float = 125.0,
        max_frequency: float = 7600.0,
        preemphasis: float = 0.97,
        preemphasis_htk_flavor: bool = True,
        fft_overdrive: bool = True,
        dither: float = 0.0,
        input_scale_factor: float = 1.0,
        mel_floor: float = 1e-5,
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

        hann_arange = np.arange(self.frame_length, dtype=np.float32)
        window = 0.5 * (1 - np.cos(2 * np.pi * hann_arange / self.frame_length))
        self.window = window.astype(np.float32)

        self.mel_filters = create_fb_matrix(
            n_freqs=self.fft_length // 2 + 1,
            f_min=min_frequency,
            f_max=max_frequency,
            n_mels=feature_size,
            sample_rate=self.sampling_rate,
            norm=None,
            fft_length=fft_length,
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

        if not is_batched:
            raw_speech = [raw_speech]
        raw_speech = [np.asarray([rs]).T for rs in raw_speech]

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

        return BatchFeature(
            {"input_features": prepared_speech, "input_features_mask": prepared_speech_mask},
            tensor_type=return_tensors,
        )


__all__ = ["Gemma3nAudioFeatureExtractor"]
