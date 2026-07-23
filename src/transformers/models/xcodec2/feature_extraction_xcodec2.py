
import copy
from typing import Any

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
from ...utils.import_utils import is_torch_available, is_torchaudio_available, requires


if is_torch_available():
    import torch
    import torch.nn.functional as F

if is_torchaudio_available():
    import torchaudio


logger = logging.get_logger(__name__)


@requires(backends=("torchaudio",))
class Xcodec2FeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features", "input_values", "padding_mask", "input_features_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        padding_value=1.0,
        hop_length=320,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.hop_length = hop_length
        self.acoustic_encoder_padder = SequenceFeatureExtractor(
            feature_size=1,
            sampling_rate=sampling_rate,
            padding_value=0.0,
        )
        self.acoustic_encoder_padder.model_input_names = ["audio", "padding_mask"]

        self.stride = 2
        self.num_mel_bins = 80
        self.frame_length = 400
        self.frame_shift = 160

    def __call__(
        self,
        audio: AudioInput,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Numpy array or torch tensor with shape (num_channels, sequence_length). A list of such arrays or
                tensors can also be provided for a batch of inputs.
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
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sample rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            device (`str`, *optional*, defaults to `"cpu"`):
                Device for PyTorch tensors during mel-filter bank feature extraction.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to the tokenizer or the feature
                extractor.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `audio` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        audio = make_list_of_audio(audio)
        for example in audio:
            if example.ndim > 2:
                raise ValueError(f"Expected input shape (channels, length) but got shape {example.shape}")
        batch_size = len(audio)

        audio = [F.pad(torch.as_tensor(a), (0, 1), value=0.0) for a in audio]
        padded_inputs = self.acoustic_encoder_padder.pad(
            BatchFeature({"audio": audio}),
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_attention_mask=padding,
            pad_to_multiple_of=self.hop_length,
            return_tensors="pt",
        )
        padding_mask = padded_inputs.pop("attention_mask")
        padded_audio = padded_inputs["audio"][:, None, :]

        mel_features = []
        for i in range(batch_size):
            orig_len = int(padding_mask[i].sum().item()) if padding_mask is not None else padded_audio.shape[-1]
            per_sample_len = ((orig_len + self.hop_length - 1) // self.hop_length) * self.hop_length
            valid_len = min(per_sample_len, padded_audio.shape[-1])
            waveform = padded_audio[i, :, :valid_len]
            waveform = F.pad(waveform, (self.hop_length // 2, self.hop_length // 2), value=0.0)
            waveform = waveform.to(device)
            features = torchaudio.compliance.kaldi.fbank(
                waveform * (2**15),
                num_mel_bins=self.num_mel_bins,
                frame_length=self.frame_length / self.sampling_rate * 1000,
                frame_shift=self.frame_shift / self.sampling_rate * 1000,
                sample_frequency=self.sampling_rate,
                window_type="povey",
                preemphasis_coefficient=0.97,
                remove_dc_offset=True,
                use_log_fbank=True,
                use_energy=False,
                dither=0.0,
                snip_edges=True,
                low_freq=20,
                high_freq=self.sampling_rate // 2,
            )
            features = (features - features.mean(0)) / torch.sqrt(features.var(0, unbiased=True) + 1e-7)
            mel_features.append(features)
        encoded_inputs = BatchFeature({"input_features": mel_features})
        padded_mel = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=self.stride,
            return_attention_mask=padding,
            return_tensors="pt",
        )
        audio_spectrogram = padded_mel["input_features"]
        spectrogram_mask = padded_mel.get("attention_mask")
        trimmed_frames = audio_spectrogram.shape[1] - (audio_spectrogram.shape[1] % self.stride)
        audio_spectrogram = audio_spectrogram[:, :trimmed_frames, :].reshape(
            batch_size, trimmed_frames // self.stride, self.num_mel_bins * self.stride
        )
        if spectrogram_mask is not None:
            spectrogram_mask = (
                spectrogram_mask[:, :trimmed_frames]
                .reshape(batch_size, trimmed_frames // self.stride, self.stride)
                .min(dim=-1)
                .values
            )

        return BatchFeature(
            {
                "input_values": padded_audio,
                "padding_mask": padding_mask,
                "input_features": audio_spectrogram,
                "input_features_mask": spectrogram_mask,
            },
            tensor_type=return_tensors,
        )

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        output.pop("acoustic_encoder_padder", None)
        return output


__all__ = ["Xcodec2FeatureExtractor"]
