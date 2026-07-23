
import copy
from typing import Any

import numpy as np

from ...audio_utils import chroma_filter_bank
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_torch_available, is_torchaudio_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    import torch

if is_torchaudio_available():
    import torchaudio

logger = logging.get_logger(__name__)


@requires(backends=("torchaudio",))
class MusicgenMelodyFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=12,
        sampling_rate=32000,
        hop_length=4096,
        chunk_length=30,
        n_fft=16384,
        num_chroma=12,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        stem_indices=[3, 2],
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
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.sampling_rate = sampling_rate
        self.chroma_filters = torch.from_numpy(
            chroma_filter_bank(sampling_rate=sampling_rate, num_frequency_bins=n_fft, tuning=0, num_chroma=num_chroma)
        ).float()
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, win_length=n_fft, hop_length=hop_length, power=2, center=True, pad=0, normalized=True
        )
        self.stem_indices = stem_indices

    def _torch_extract_fbank_features(self, waveform: torch.Tensor) -> torch.Tensor:
        pass

    def _extract_stem_indices(self, audio, sampling_rate=None):
        pass

    def __call__(
        self,
        audio: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: bool = True,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        padding: str | None = True,
        max_length: int | None = None,
        sampling_rate: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            audio (`torch.Tensor`, `np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a torch tensor, a numpy array, a list of float
                values, a list of numpy arrays, a list of torch tensors, or a list of list of float values.
                If `audio` is the output of Demucs, it has to be a torch tensor of shape `(batch_size, num_stems, channel_size, audio_length)`.
                Otherwise, it must be mono or stereo channel audio.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>
                For Musicgen Melody models, audio `attention_mask` is not necessary.
                </Tip>

            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
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
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
                Note that if `audio` is the output of Demucs, `sampling_rate` must be the sampling rate at which Demucs operates.
        """

        if sampling_rate is None:
            logger.warning_once(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if isinstance(audio, torch.Tensor) and len(audio.shape) == 4:
            logger.warning_once(
                "`audio` is a 4-dimensional torch tensor and has thus been recognized as the output of `Demucs`. "
                "If this is not the case, make sure to read Musicgen Melody docstrings and "
                "to correct `audio` to get the right behaviour."
                "Link to the docstrings: https://huggingface.co/docs/transformers/main/en/model_doc/musicgen_melody"
            )
            audio = self._extract_stem_indices(audio, sampling_rate=sampling_rate)
        elif sampling_rate is not None and sampling_rate != self.sampling_rate:
            audio = torchaudio.functional.resample(
                audio, sampling_rate, self.sampling_rate, rolloff=0.945, lowpass_filter_width=24
            )

        is_batched = isinstance(audio, (np.ndarray, torch.Tensor)) and len(audio.shape) > 1
        is_batched = is_batched or (
            isinstance(audio, (list, tuple)) and (isinstance(audio[0], (torch.Tensor, np.ndarray, tuple, list)))
        )

        if is_batched and not isinstance(audio[0], torch.Tensor):
            audio = [torch.tensor(speech, dtype=torch.float32).unsqueeze(-1) for speech in audio]
        elif is_batched:
            audio = [speech.unsqueeze(-1) for speech in audio]
        elif not is_batched and not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(-1)

        if isinstance(audio[0], torch.Tensor) and audio[0].dtype is torch.float64:
            audio = [speech.to(torch.float32) for speech in audio]

        if not is_batched:
            audio = [audio]

        if len(audio[0].shape) == 3:
            logger.warning_once(
                "`audio` has been detected as a batch of stereo signals. Will be convert to mono signals. "
                "If this is an undesired behaviour, make sure to read Musicgen Melody docstrings and "
                "to correct `audio` to get the right behaviour."
                "Link to the docstrings: https://huggingface.co/docs/transformers/main/en/model_doc/musicgen_melody"
            )
            audio = [stereo.mean(dim=0) for stereo in audio]

        batched_speech = BatchFeature({"input_features": audio})

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length if max_length else self.n_samples,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors="pt",
        )

        input_features = self._torch_extract_fbank_features(padded_inputs["input_features"].squeeze(-1))

        padded_inputs["input_features"] = input_features

        if return_attention_mask:
            padded_inputs["attention_mask"] = padded_inputs["attention_mask"][:, :: self.hop_length]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        if "mel_filters" in output:
            del output["mel_filters"]
        if "window" in output:
            del output["window"]
        if "chroma_filters" in output:
            del output["chroma_filters"]
        if "spectrogram" in output:
            del output["spectrogram"]
        return output


__all__ = ["MusicgenMelodyFeatureExtractor"]
