
import numpy as np

from ...audio_utils import mel_filter_bank
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import logging
from ...utils.import_utils import is_torch_available, requires


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


@requires(backends=("torch",))
class Qwen3ASRFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=True,
        n_window=50,
        min_length=8000,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_length = min_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.dither = dither
        self.n_window = n_window
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def _torch_extract_fbank_features(self, waveform: np.ndarray, device: str = "cpu") -> np.ndarray:
        pass

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | None = "pt",
        return_attention_mask: bool | None = None,
        padding: str | None = "max_length",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        n_window: int | None = None,
        device: str | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
        r"""
        Prepare log-mel features from one or several audio sequences.

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Mono-channel audio only.
            pad_to_multiple_of (`int`, *optional*):
                If set, pads the raw audio to a multiple of this value (in samples). Separate from
                ``n_window``, which applies to the mel-frame axis.
            n_window (`int`, *optional*):
                Override the instance's ``n_window`` for this call. The mel axis is padded to a multiple
                of ``2 * n_window``. Set to ``0`` to skip mel-axis padding entirely.
            device (`str`, *optional*, defaults to `"cpu"`):
                Device used to compute the log-mel spectrogram.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        if self.min_length > 0:
            raw_speech = [
                np.pad(s, ((0, self.min_length - s.shape[0]), (0, 0))) if s.shape[0] < self.min_length else s
                for s in raw_speech
            ]

        batched_speech = BatchFeature({"input_features": raw_speech})

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length if max_length else self.n_samples,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        input_features = padded_inputs["input_features"].transpose(2, 0, 1)
        input_features = self._torch_extract_fbank_features(input_features[0], device)
        padded_inputs["input_features"] = input_features

        rescaled_attention_mask = padded_inputs["attention_mask"][:, :: self.hop_length]
        if padded_inputs["attention_mask"].shape[1] % self.hop_length != 0:
            rescaled_attention_mask = rescaled_attention_mask[:, :-1]
        padded_inputs["attention_mask"] = rescaled_attention_mask

        if n_window is None:
            n_window = self.n_window
        multiple = n_window * 2
        if multiple and multiple > 1:
            remainder = padded_inputs["input_features"].shape[-1] % multiple
            pad = (multiple - remainder) if remainder else 0
            if pad:
                padded_inputs["input_features"] = np.pad(padded_inputs["input_features"], [(0, 0), (0, 0), (0, pad)])
                padded_inputs["attention_mask"] = np.pad(padded_inputs["attention_mask"], [(0, 0), (0, pad)])

        if not return_attention_mask:
            padded_inputs.pop("attention_mask", None)

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


__all__ = ["Qwen3ASRFeatureExtractor"]
