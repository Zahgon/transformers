
import warnings

import numpy
import numpy as np

from ...audio_utils import mel_filter_bank, spectrogram
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import (
    TensorType,
    is_essentia_available,
    is_librosa_available,
    is_scipy_available,
    logging,
    requires_backends,
)
from ...utils.import_utils import requires


if is_essentia_available():
    import essentia.standard

if is_librosa_available():
    import librosa

if is_scipy_available():
    import scipy


logger = logging.get_logger(__name__)


@requires(backends=("essentia", "librosa", "scipy", "torch"))
class Pop2PianoFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features", "beatsteps", "extrapolated_beatstep"]

    def __init__(
        self,
        sampling_rate: int = 22050,
        padding_value: int = 0,
        window_size: int = 4096,
        hop_length: int = 1024,
        min_frequency: float = 10.0,
        feature_size: int = 512,
        num_bars: int = 2,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.window_size = window_size
        self.hop_length = hop_length
        self.min_frequency = min_frequency
        self.feature_size = feature_size
        self.num_bars = num_bars
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=(self.window_size // 2) + 1,
            num_mel_filters=self.feature_size,
            min_frequency=self.min_frequency,
            max_frequency=float(self.sampling_rate // 2),
            sampling_rate=self.sampling_rate,
            norm=None,
            mel_scale="htk",
        )

    def mel_spectrogram(self, sequence: np.ndarray):
        pass

    def extract_rhythm(self, audio: np.ndarray):
        pass

    def interpolate_beat_times(
        self, beat_times: numpy.ndarray, steps_per_beat: numpy.ndarray, n_extend: numpy.ndarray
    ):
        pass

    def preprocess_mel(self, audio: np.ndarray, beatstep: np.ndarray):
        pass

    def _pad(self, features: np.ndarray, add_zero_line=True):
        features_shapes = [each_feature.shape for each_feature in features]
        attention_masks, padded_features = [], []
        for i, each_feature in enumerate(features):
            if len(each_feature.shape) == 3:
                features_pad_value = max([*zip(*features_shapes)][1]) - features_shapes[i][1]
                attention_mask = np.ones(features_shapes[i][:2], dtype=np.int64)
                feature_padding = ((0, 0), (0, features_pad_value), (0, 0))
                attention_mask_padding = (feature_padding[0], feature_padding[1])

            else:
                each_feature = each_feature.reshape(1, -1)
                features_pad_value = max([*zip(*features_shapes)][0]) - features_shapes[i][0]
                attention_mask = np.ones(features_shapes[i], dtype=np.int64).reshape(1, -1)
                feature_padding = attention_mask_padding = ((0, 0), (0, features_pad_value))

            each_padded_feature = np.pad(each_feature, feature_padding, "constant", constant_values=self.padding_value)
            attention_mask = np.pad(
                attention_mask, attention_mask_padding, "constant", constant_values=self.padding_value
            )

            if add_zero_line:
                zero_array_len = max([*zip(*features_shapes)][1])

                each_padded_feature = np.concatenate(
                    [each_padded_feature, np.zeros([1, zero_array_len, self.feature_size])], axis=0
                )
                attention_mask = np.concatenate(
                    [attention_mask, np.zeros([1, zero_array_len], dtype=attention_mask.dtype)], axis=0
                )

            padded_features.append(each_padded_feature)
            attention_masks.append(attention_mask)

        padded_features = np.concatenate(padded_features, axis=0).astype(np.float32)
        attention_masks = np.concatenate(attention_masks, axis=0).astype(np.int64)

        return padded_features, attention_masks

    def pad(
        self,
        inputs: BatchFeature,
        is_batched: bool,
        return_attention_mask: bool,
        return_tensors: str | TensorType | None = None,
    ):
        """
        Pads the inputs to same length and returns attention_mask.

        Args:
            inputs (`BatchFeature`):
                Processed audio features.
            is_batched (`bool`):
                Whether inputs are batched or not.
            return_attention_mask (`bool`):
                Whether to return attention mask or not.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
                If nothing is specified, it will return list of `np.ndarray` arrays.
        Return:
            `BatchFeature` with attention_mask, attention_mask_beatsteps and attention_mask_extrapolated_beatstep added
            to it:
            - **attention_mask** numpy.ndarray of shape `(batch_size, max_input_features_seq_length)` --
                Example :
                    1, 1, 1, 0, 0 (audio 1, also here it is padded to max length of 5 that's why there are 2 zeros at
                    the end indicating they are padded)

                    0, 0, 0, 0, 0 (zero pad to separate audio 1 and 2)

                    1, 1, 1, 1, 1 (audio 2)

                    0, 0, 0, 0, 0 (zero pad to separate audio 2 and 3)

                    1, 1, 1, 1, 1 (audio 3)
            - **attention_mask_beatsteps** numpy.ndarray of shape `(batch_size, max_beatsteps_seq_length)`
            - **attention_mask_extrapolated_beatstep** numpy.ndarray of shape `(batch_size,
              max_extrapolated_beatstep_seq_length)`
        """

        processed_features_dict = {}
        for feature_name, feature_value in inputs.items():
            if feature_name == "input_features":
                padded_feature_values, attention_mask = self._pad(feature_value, add_zero_line=True)
                processed_features_dict[feature_name] = padded_feature_values
                if return_attention_mask:
                    processed_features_dict["attention_mask"] = attention_mask
            else:
                padded_feature_values, attention_mask = self._pad(feature_value, add_zero_line=False)
                processed_features_dict[feature_name] = padded_feature_values
                if return_attention_mask:
                    processed_features_dict[f"attention_mask_{feature_name}"] = attention_mask

        if not is_batched and not return_attention_mask:
            processed_features_dict["input_features"] = processed_features_dict["input_features"][:-1, ...]

        outputs = BatchFeature(processed_features_dict, tensor_type=return_tensors)

        return outputs

    def __call__(
        self,
        audio: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        sampling_rate: int | list[int],
        steps_per_beat: int = 2,
        resample: bool | None = True,
        return_attention_mask: bool | None = False,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model.

        Args:
            audio (`np.ndarray`, `List`):
                The audio or batch of audio to be processed. Each audio can be a numpy array, a list of float values, a
                list of numpy arrays or a list of list of float values.
            sampling_rate (`int`):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            steps_per_beat (`int`, *optional*, defaults to 2):
                This is used in interpolating `beat_times`.
            resample (`bool`, *optional*, defaults to `True`):
                Determines whether to resample the audio to `sampling_rate` or not before processing. Must be True
                during inference.
            return_attention_mask (`bool` *optional*, defaults to `False`):
                Denotes if attention_mask for input_features, beatsteps and extrapolated_beatstep will be given as
                output or not. Automatically set to True for batched inputs.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
                If nothing is specified, it will return list of `np.ndarray` arrays.
        """

        requires_backends(self, ["librosa"])
        is_batched = isinstance(audio, (list, tuple)) and isinstance(audio[0], (np.ndarray, tuple, list))
        if is_batched:
            if not isinstance(sampling_rate, list):
                raise ValueError(
                    "Please give sampling_rate of each audio separately when you are passing multiple raw_audios at the same time. "
                    f"Received {sampling_rate}, expected [audio_1_sr, ..., audio_n_sr]."
                )
            return_attention_mask = True if return_attention_mask is None else return_attention_mask
        else:
            audio = [audio]
            sampling_rate = [sampling_rate]
            return_attention_mask = False if return_attention_mask is None else return_attention_mask

        batch_input_features, batch_beatsteps, batch_ext_beatstep = [], [], []
        for single_raw_audio, single_sampling_rate in zip(audio, sampling_rate):
            bpm, beat_times, confidence, estimates, essentia_beat_intervals = self.extract_rhythm(
                audio=single_raw_audio
            )
            beatsteps = self.interpolate_beat_times(beat_times=beat_times, steps_per_beat=steps_per_beat, n_extend=1)

            if self.sampling_rate != single_sampling_rate and self.sampling_rate is not None:
                if resample:
                    single_raw_audio = librosa.core.resample(
                        single_raw_audio,
                        orig_sr=single_sampling_rate,
                        target_sr=self.sampling_rate,
                        res_type="kaiser_best",
                    )
                else:
                    warnings.warn(
                        f"The sampling_rate of the provided audio is different from the target sampling_rate "
                        f"of the Feature Extractor, {self.sampling_rate} vs {single_sampling_rate}. "
                        f"In these cases it is recommended to use `resample=True` in the `__call__` method to "
                        f"get the optimal behaviour."
                    )

            single_sampling_rate = self.sampling_rate
            start_sample = int(beatsteps[0] * single_sampling_rate)
            end_sample = int(beatsteps[-1] * single_sampling_rate)

            input_features, extrapolated_beatstep = self.preprocess_mel(
                single_raw_audio[start_sample:end_sample], beatsteps - beatsteps[0]
            )

            mel_specs = self.mel_spectrogram(input_features.astype(np.float32))

            log_mel_specs = np.log(np.clip(mel_specs, a_min=1e-6, a_max=None))

            input_features = np.transpose(log_mel_specs, (0, -1, -2))

            batch_input_features.append(input_features)
            batch_beatsteps.append(beatsteps)
            batch_ext_beatstep.append(extrapolated_beatstep)

        output = BatchFeature(
            {
                "input_features": batch_input_features,
                "beatsteps": batch_beatsteps,
                "extrapolated_beatstep": batch_ext_beatstep,
            }
        )

        output = self.pad(
            output,
            is_batched=is_batched,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )

        return output


__all__ = ["Pop2PianoFeatureExtractor"]
