
import math
from collections.abc import Sequence

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils_base import AudioInput
from ...utils import is_torch_available, is_torchaudio_available, logging
from ...utils.import_utils import requires_backends


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torchaudio_available():
    import torchaudio


class GraniteSpeechFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_features"]

    def __init__(
        self,
        sampling_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        projector_window_size: int = 15,
        projector_downsample_rate: int = 5,
        padding_value: float = 0.0,
        **kwargs,
    ):
        feature_size = n_mels if kwargs.get("feature_size") is None else kwargs.pop("feature_size")
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.sampling_rate = sampling_rate
        self.melspec_kwargs = {
            "sample_rate": sampling_rate,
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "n_mels": n_mels,
        }
        requires_backends(self, ["torchaudio"])
        self.mel_filters = torchaudio.transforms.MelSpectrogram(**self.melspec_kwargs)
        self.projector_window_size = projector_window_size
        self.projector_downsample_rate = projector_downsample_rate

    def __call__(
        self,
        audios: AudioInput,
        device: str | None = "cpu",
    ) -> BatchFeature:
        requires_backends(self, ["torchaudio"])

        speech_inputs = {}
        batched_audio, audio_lengths = self._get_audios_and_audio_lengths(audios)
        speech_inputs["input_features"] = self._extract_mel_spectrograms(
            batched_audio,
            device=device,
        )
        audio_embed_sizes = self._get_num_audio_features(audio_lengths)
        speech_inputs["audio_embed_sizes"] = audio_embed_sizes
        speech_inputs["input_features_mask"] = torch.arange(max(audio_embed_sizes)).view(1, -1) < torch.tensor(
            audio_embed_sizes
        ).view(-1, 1)
        return BatchFeature(data=speech_inputs)

    def _extract_mel_spectrograms(self, audio: "torch.Tensor", device="cpu"):
        pass

    def _get_num_audio_features(self, audio_lengths: Sequence[int]) -> Sequence[int]:
        pass

    def _get_audios_and_audio_lengths(self, audios: AudioInput) -> Sequence["torch.Tensor", Sequence[int]]:
        pass


__all__ = ["GraniteSpeechFeatureExtractor"]
