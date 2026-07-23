

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...image_processing_utils import BatchFeature
from ...utils import (
    TensorType,
    is_torch_available,
)


if is_torch_available():
    import torch


class Gemma4UnifiedAudioFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 640,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        audio_samples_per_token: int = 640,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.audio_samples_per_token = audio_samples_per_token

    def _extract_waveform_features(
        self,
        waveform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str = "longest",
        max_length: int | None = None,
        truncation: bool = True,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Chunk raw audio waveforms into fixed-length frames for the unified model.

        Args:
            raw_speech:
                The raw audio waveform(s) to process.
            padding (`str`, *optional*, defaults to `"longest"`):
                Padding strategy for batches with different lengths.
            max_length (`int`, *optional*):
                Maximum number of tokens to produce per audio.
            truncation (`bool`, *optional*, defaults to `True`):
                Whether to truncate audio above `max_length` tokens.
            return_tensors (`str`, *optional*):
                The type of tensors to return.
        """
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
            raw_speech = [raw_speech]
        elif not isinstance(raw_speech, (list, tuple)):
            raw_speech = [np.asarray(raw_speech)]
        else:
            raw_speech = [np.asarray(s) for s in raw_speech]

        all_features = [{"input_features": self._extract_waveform_features(waveform)[0]} for waveform in raw_speech]

        padded_inputs = self.pad(
            all_features,
            padding=padding,
            max_length=max_length,
            truncation=truncation and max_length is not None,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )

        mask = padded_inputs.pop("attention_mask")
        if is_torch_available() and isinstance(mask, torch.Tensor):
            mask = mask.bool()
        else:
            mask = np.asarray(mask, dtype=bool)
        padded_inputs["input_features_mask"] = mask

        return padded_inputs


__all__ = ["Gemma4UnifiedAudioFeatureExtractor"]
