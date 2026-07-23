
from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, logging
from ...utils.import_utils import is_torch_available, requires


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


@requires(backends=("torch",))
class VibeVoiceAcousticTokenizerFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=24000,
        padding_value=0.0,
        normalize_audio=True,
        target_dB_FS=-25,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.normalize_audio = normalize_audio
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def __call__(
        self,
        audio: AudioInput,
        sampling_rate: int | None = None,
        padding: bool | str | PaddingStrategy | None = True,
        pad_to_multiple_of: int | None = None,
        max_length: int | None = None,
        return_attention_mask: bool | None = True,
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`:
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a torch tensor,
                a list of numpy arrays or a list of torch tensors.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

        """
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

        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        audio = make_list_of_audio(audio)
        for idx, example in enumerate(audio):
            example = torch.tensor(example, dtype=torch.float32)
            if example.ndim != 1:
                raise ValueError(f"Audio should be mono, got shape: {example.shape}")
            audio[idx] = example

        if self.normalize_audio:
            for idx, example in enumerate(audio):
                rms = torch.sqrt(torch.mean(example**2))
                example *= 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
                max_val = torch.max(torch.abs(example))
                if max_val > 1.0:
                    example = example / (max_val + self.eps)
                audio[idx] = example

        output_values = BatchFeature({"input_values": audio})
        if padding or pad_to_multiple_of:
            output_values = self.pad(
                output_values,
                padding=padding,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                max_length=max_length,
            )
            if return_attention_mask:
                output_values["padding_mask"] = output_values.pop("attention_mask")

        output_values["input_values"] = output_values["input_values"][:, None, :]

        return output_values


__all__ = ["VibeVoiceAcousticTokenizerFeatureExtractor"]
