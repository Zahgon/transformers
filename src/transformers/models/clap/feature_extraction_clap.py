
import copy
from typing import Any

import numpy as np

from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_torch_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


@requires(backends=("torch",))
class ClapFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features", "is_longer"]

    def __init__(
        self,
        feature_size=64,
        sampling_rate=48_000,
        hop_length=480,
        max_length_s=10,
        fft_window_size=1024,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        frequency_min: float = 0,
        frequency_max: float = 14_000,
        top_db: int | None = None,
        truncation: str = "fusion",
        padding: str = "repeatpad",
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.top_db = top_db
        self.truncation = truncation
        self.padding = padding
        self.fft_window_size = fft_window_size
        self.nb_frequency_bins = (fft_window_size >> 1) + 1
        self.hop_length = hop_length
        self.max_length_s = max_length_s
        self.nb_max_samples = max_length_s * sampling_rate
        self.sampling_rate = sampling_rate
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.nb_frequency_bins,
            num_mel_filters=feature_size,
            min_frequency=frequency_min,
            max_frequency=frequency_max,
            sampling_rate=sampling_rate,
            norm=None,
            mel_scale="htk",
        )
        self.mel_filters_slaney = mel_filter_bank(
            num_frequency_bins=self.nb_frequency_bins,
            num_mel_filters=feature_size,
            min_frequency=frequency_min,
            max_frequency=frequency_max,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance, except for the
            mel filter banks, which do not need to be saved or printed as they are too long.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        if "mel_filters" in output:
            del output["mel_filters"]
        if "mel_filters_slaney" in output:
            del output["mel_filters_slaney"]
        return output

    def _np_extract_fbank_features(self, waveform: np.ndarray, mel_filters: np.ndarray | None = None) -> np.ndarray:
        pass

    def _random_mel_fusion(self, mel, total_frames, chunk_frames):
        pass

    def _get_input_mel(self, waveform: np.ndarray, max_length, truncation, padding) -> np.ndarray:
        pass

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: str | None = None,
        padding: str | None = None,
        max_length: int | None = None,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            truncation (`str`, *optional*):
                Truncation pattern for long audio inputs. Two patterns are available:
                    - `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and
                      a downsampled version of the entire mel spectrogram.
                If `config.fusion` is set to True, shorter audios also need to return 4 mels, which will just be a
                copy of the original mel obtained from the padded audio.
                    - `rand_trunc` will select a random crop of the mel spectrogram.
            padding (`str`, *optional*):
               Padding pattern for shorter audio inputs. Three patterns were originally implemented:
                    - `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.
                    - `repeat`: the audio is repeated and then cut to fit the `max_length`
                    - `pad`: the audio is padded.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.np.array` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
        """
        truncation = truncation if truncation is not None else self.truncation
        padding = padding if padding else self.padding

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
            raw_speech = [np.asarray(speech, dtype=np.float64) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float64)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float64)

        if not is_batched:
            raw_speech = [np.asarray(raw_speech)]

        padded_inputs = [
            self._get_input_mel(waveform, max_length if max_length else self.nb_max_samples, truncation, padding)
            for waveform in raw_speech
        ]

        input_mel = []
        is_longer = []
        for mel, longer in padded_inputs:
            input_mel.append(mel)
            is_longer.append(longer)

        if truncation == "fusion" and sum(is_longer) == 0:
            rand_idx = np.random.randint(0, len(input_mel))
            is_longer[rand_idx] = True

        if isinstance(input_mel[0], list):
            input_mel = [np.asarray(feature, dtype=np.float64) for feature in input_mel]

        is_longer = [[longer] for longer in is_longer]

        input_features = {"input_features": input_mel, "is_longer": is_longer}
        input_features = BatchFeature(input_features)

        if return_tensors is not None:
            input_features = input_features.convert_to_tensors(return_tensors)

        return input_features


__all__ = ["ClapFeatureExtractor"]
