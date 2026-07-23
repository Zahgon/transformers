
from typing import Any

import numpy as np

from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


class SpeechT5FeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        do_normalize: bool = False,
        num_mel_bins: int = 80,
        hop_length: int = 16,
        win_length: int = 64,
        win_function: str = "hann_window",
        fmin: float = 80,
        fmax: float = 7600,
        mel_floor: float = 1e-10,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.do_normalize = do_normalize
        self.return_attention_mask = return_attention_mask

        self.num_mel_bins = num_mel_bins
        self.hop_length = hop_length
        self.win_length = win_length
        self.win_function = win_function
        self.fmin = fmin
        self.fmax = fmax
        self.mel_floor = mel_floor

        self.sample_size = win_length * sampling_rate // 1000
        self.sample_stride = hop_length * sampling_rate // 1000
        self.n_fft = optimal_fft_length(self.sample_size)
        self.n_freqs = (self.n_fft // 2) + 1

        self.window = window_function(window_length=self.sample_size, name=self.win_function, periodic=True)

        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_freqs,
            num_mel_filters=self.num_mel_bins,
            min_frequency=self.fmin,
            max_frequency=self.fmax,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: list[np.ndarray], attention_mask: list[np.ndarray], padding_value: float = 0.0
    ) -> list[np.ndarray]:
        pass

    def _extract_mel_features(
        self,
        one_waveform: np.ndarray,
    ) -> np.ndarray:
        pass

    def __call__(
        self,
        audio: np.ndarray | list[float] | list[np.ndarray] | list[list[float]] | None = None,
        audio_target: np.ndarray | list[float] | list[np.ndarray] | list[list[float]] | None = None,
        padding: bool | str | PaddingStrategy = False,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Pass in a value for `audio` to extract waveform features. Pass in a value for `audio_target` to extract log-mel
        spectrogram features.

        Args:
            audio (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`, *optional*):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. This outputs waveform features. Must
                be mono channel audio, not stereo, i.e. single float per timestep.
            audio_target (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`, *optional*):
                The sequence or batch of sequences to be processed as targets. Each sequence can be a numpy array, a
                list of float values, a list of numpy arrays or a list of list of float values. This outputs log-mel
                spectrogram features.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` or `audio_target` input was sampled. It is strongly recommended
                to pass `sampling_rate` at the forward call to prevent silent errors.
        """
        if audio is None and audio_target is None:
            raise ValueError("You must provide either `audio` or `audio_target` values.")

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided audio input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if audio is not None:
            inputs = self._process_audio(
                audio,
                False,
                padding,
                max_length,
                truncation,
                pad_to_multiple_of,
                return_attention_mask,
                return_tensors,
                **kwargs,
            )
        else:
            inputs = None

        if audio_target is not None:
            inputs_target = self._process_audio(
                audio_target,
                True,
                padding,
                max_length,
                truncation,
                pad_to_multiple_of,
                return_attention_mask,
                return_tensors,
                **kwargs,
            )

            if inputs is None:
                return inputs_target
            else:
                inputs["labels"] = inputs_target["input_values"]
                decoder_attention_mask = inputs_target.get("attention_mask")
                if decoder_attention_mask is not None:
                    inputs["decoder_attention_mask"] = decoder_attention_mask

        return inputs

    def _process_audio(
        self,
        speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        is_target: bool = False,
        padding: bool | str | PaddingStrategy = False,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        pass

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()

        names = ["window", "mel_filters", "sample_size", "sample_stride", "n_fft", "n_freqs"]
        for name in names:
            if name in output:
                del output[name]

        return output


__all__ = ["SpeechT5FeatureExtractor"]
