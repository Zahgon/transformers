
import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_librosa_available, is_torch_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    import torch

if is_librosa_available():
    import librosa


EPSILON = 1e-5
LOG_ZERO_GUARD_VALUE = 2**-24


logger = logging.get_logger(__name__)


@requires(backends=("torch", "librosa"))
class CohereAsrFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=512,
        win_length=400,
        preemphasis=0.97,
        padding_value=0.0,
        dither=1e-5,
        max_audio_clip_s=35.0,
        overlap_chunk_second=5.0,
        min_energy_window_samples=1600,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = preemphasis
        self.dither = dither
        self.max_audio_clip_s = max_audio_clip_s
        self.overlap_chunk_second = overlap_chunk_second
        self.min_energy_window_samples = min_energy_window_samples

        mel_filters = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=feature_size, fmin=0.0, fmax=sampling_rate / 2, norm="slaney"
        )
        self.mel_filters = torch.from_numpy(mel_filters).to(torch.float32)

    def _find_split_point_energy(self, waveform: torch.Tensor, start_idx: int, end_idx: int) -> int:
        pass

    def _split_audio_chunks_energy(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        pass

    def _apply_dither(self, waveform: torch.Tensor, audio_lengths: torch.Tensor) -> torch.Tensor:
        pass

    def _torch_extract_fbank_features(self, waveform, device="cpu"):
        pass

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        padding: str | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        do_normalize: bool | None = None,
        device: str | None = "cpu",
        return_token_timestamps: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s). Implementation uses PyTorch for
        the STFT computation if available, otherwise a slower NumPy based one.

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For CohereAsr models, `attention_mask` should always be passed for batched inference, to avoid subtle
                bugs.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            padding_value (`float`, *optional*, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
            do_normalize (`bool`, *optional*, defaults to `False`):
                Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
                improve the performance of the model.
            device (`str`, *optional*, defaults to `'cpu'`):
                Specifies the device for computation of the log-mel spectrogram of audio signals in the
                `_torch_extract_fbank_features` method. (e.g., "cpu", "cuda")
            return_token_timestamps (`bool`, *optional*, defaults to `None`):
                Deprecated. Use `return_attention_mask` instead from which the number of frames can be inferred.

                Whether or not to return the number of frames of the input raw_speech.
                These num_frames can be used by the model to compute word level timestamps.
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

        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.tensor(raw_speech)
        elif isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], np.ndarray):
            raw_speech = [torch.tensor(speech) for speech in raw_speech]

        is_batched_torch = isinstance(raw_speech, torch.Tensor) and len(raw_speech.shape) > 1
        if is_batched_torch and len(raw_speech.shape) > 2:
            logger.warning(
                f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                "We will take the mean of the channels to convert to mono."
            )
            raw_speech = raw_speech.mean(-1)

        is_batched_sequence = isinstance(raw_speech, (list, tuple))
        if is_batched_sequence:
            for speech in raw_speech:
                if len(speech.shape) > 1:
                    logger.warning(
                        f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                        "We will take the mean of the channels to convert to mono."
                    )
                    speech = speech.mean(-1)

        if is_batched_torch or is_batched_sequence:
            raw_speech = [speech.to(torch.float32) for speech in raw_speech]
        else:
            raw_speech = [raw_speech.to(torch.float32)]

        fast_path_threshold_s = max(0.0, self.max_audio_clip_s - self.overlap_chunk_second)
        audio_chunk_index: list[tuple[int, int | None]] = []
        chunked_speech: list[torch.Tensor] = []
        for sample_idx, speech in enumerate(raw_speech):
            duration_s = speech.shape[0] / self.sampling_rate
            if duration_s <= fast_path_threshold_s:
                chunked_speech.append(speech)
                audio_chunk_index.append((sample_idx, None))
            else:
                chunks = self._split_audio_chunks_energy(speech)
                for chunk_idx, chunk in enumerate(chunks):
                    chunked_speech.append(chunk)
                    audio_chunk_index.append((sample_idx, chunk_idx))

        raw_speech = [speech[:, None] for speech in chunked_speech]

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
        input_features = padded_inputs.input_features.squeeze(-1)

        input_features = self._apply_dither(input_features, padded_inputs.audio_lengths)

        if self.preemphasis is not None:
            timemask = torch.arange(input_features.shape[1], device=input_features.device).unsqueeze(
                0
            ) < padded_inputs.audio_lengths.unsqueeze(1)
            input_features = torch.cat(
                [input_features[:, :1], input_features[:, 1:] - self.preemphasis * input_features[:, :-1]], dim=1
            )
            input_features = input_features.masked_fill(~timemask, 0.0)

        input_features = self._torch_extract_fbank_features(input_features, device)
        features_lengths = torch.floor_divide(
            padded_inputs.audio_lengths + self.n_fft // 2 * 2 - self.n_fft, self.hop_length
        )
        attention_mask = torch.arange(input_features.shape[1], device=device)[None, :] < features_lengths[:, None]

        mask = attention_mask.unsqueeze(-1)
        input_features_masked = input_features * mask
        mean = input_features_masked.sum(dim=1) / features_lengths.unsqueeze(-1)
        mean = mean.unsqueeze(1)
        variance = ((input_features_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        input_features = (input_features - mean) / (std + EPSILON)
        input_features *= mask

        result = BatchFeature(
            data={
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )
        result["audio_chunk_index"] = audio_chunk_index
        return result


__all__ = ["CohereAsrFeatureExtractor"]
