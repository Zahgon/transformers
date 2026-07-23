import math

import numpy as np
import torch
import torch.nn.functional as F

from ...audio_utils import mel_filter_bank
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)


def _to_exact_int(value: float, name: str, tolerance: float = 1e-6) -> int:
    rounded = round(value)
    if abs(value - rounded) > tolerance:
        raise ValueError(f"{name} must resolve to an integer sample count, got {value}")
    return int(rounded)


@requires(backends=("torch",))
class InklingFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        audio_token_duration_s: float = 0.05,
        window_size_multiplier: float = 2.0,
        n_fft: int | None = None,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.audio_token_duration_s = audio_token_duration_s
        self.window_size_multiplier = window_size_multiplier

        self.hop_length = _to_exact_int(
            audio_token_duration_s * sampling_rate, "audio_token_duration_s * sampling_rate"
        )
        self.window_size = _to_exact_int(
            audio_token_duration_s * window_size_multiplier * sampling_rate,
            "audio_token_duration_s * window_size_multiplier * sampling_rate",
        )
        self.n_fft = n_fft or self.window_size
        if self.hop_length <= 0 or self.window_size <= 0 or self.n_fft <= 0:
            raise ValueError("hop_length, window_size, and n_fft must all be positive")

        self.window = torch.hann_window(self.window_size, periodic=True, dtype=torch.float32)
        mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.mel_filters = torch.from_numpy(np.ascontiguousarray(mel_filters.T, dtype=np.float32))

    def _torch_extract_fbank_features(self, waveform: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        pass

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        sampling_rate: int | None = None,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = True,
        return_tensors: str | TensorType | None = None,
        device: str | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Extract log-mel spectrogram features from one or several audio clip(s).

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list
                of float values, a list of numpy arrays or a list of list of float values. Must be mono
                channel audio at `self.sampling_rate`, not stereo, i.e. single float per timestep. Decoding
                and resampling of raw audio (bytes / paths / URLs) is handled upstream by the processor's
                `apply_chat_template`, not here.
            sampling_rate (`int`, *optional*):
                The sampling rate of `raw_speech`, used only to validate against `self.sampling_rate`.
            device (`str`, *optional*, defaults to `"cpu"`):
                The device on which the log-mel spectrogram is computed in `_torch_extract_fbank_features`.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor was trained using a sampling "
                    f"rate of {self.sampling_rate}. Please make sure that the provided audio input "
                    f"was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning_once(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        cls_name = self.__class__.__name__

        def _to_mono(clip: "np.ndarray | torch.Tensor | list") -> torch.Tensor:
            pass

        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.from_numpy(raw_speech)
        if isinstance(raw_speech, torch.Tensor):
            if raw_speech.ndim > 2:
                raise ValueError(
                    f"A single array input must be 1-D (mono) or 2-D (multichannel); got {raw_speech.ndim} dims. "
                    "Pass a list of arrays for a batch of clips."
                )
            clips = [raw_speech]
        elif isinstance(raw_speech, (list, tuple)):
            if len(raw_speech) == 0:
                raise ValueError("Received an empty audio input.")
            if isinstance(raw_speech[0], (int, float, np.integer, np.floating)):
                clips = [raw_speech]
            else:
                clips = list(raw_speech)
        else:
            raise TypeError(f"Unsupported audio input type for {cls_name}: {type(raw_speech)}")

        raw_speech = [_to_mono(clip)[:, None] for clip in clips]

        audio_lengths = [len(speech) for speech in raw_speech]
        batched_speech = BatchFeature({"input_features": raw_speech, "audio_lengths": audio_lengths})
        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        input_waveforms = padded_inputs.input_features.squeeze(-1)  # (batch_size, num_samples)

        input_features = self._torch_extract_fbank_features(input_waveforms, device)  # (batch_size, T, feature_size)

        num_frames = torch.div(
            padded_inputs.audio_lengths + self.hop_length - 1, self.hop_length, rounding_mode="floor"
        )
        input_features_mask = torch.arange(input_features.shape[1], device=device)[None, :] < num_frames[:, None]
        input_features = input_features * input_features_mask.unsqueeze(-1)

        data = {"input_features": input_features}
        if return_attention_mask:
            data["input_features_mask"] = input_features_mask
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["InklingFeatureExtractor"]
