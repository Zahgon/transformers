
import re
from itertools import islice
from pathlib import Path

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_soundfile_available, is_torch_available, logging
from ...utils.import_utils import requires, requires_backends


if is_torch_available():
    import torch
    import torch.nn.functional as F


if is_soundfile_available():
    import soundfile as sf


logger = logging.get_logger(__name__)


class HiggsAudioV2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "padding": False,
            "sampling_rate": 24000,
        },
    }


@requires(backends=("torch",))
class HiggsAudioV2Processor(ProcessorMixin):

    feature_extractor_class = "DacFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "HiggsAudioV2TokenizerModel"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_tokenizer,
        chat_template=None,
        audio_token="<|AUDIO_OUT|>",
        audio_bos_token="<|audio_out_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_delay_token="<|reserved_special_token_6|>",
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
    ):
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        self.audio_bos_token = tokenizer.audio_bos_token if hasattr(tokenizer, "audio_bos_token") else audio_bos_token
        self.audio_eos_token = tokenizer.audio_eos_token if hasattr(tokenizer, "audio_eos_token") else audio_eos_token
        self.audio_delay_token = (
            tokenizer.audio_delay_token if hasattr(tokenizer, "audio_delay_token") else audio_delay_token
        )
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token_id = tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids(self.audio_eos_token)
        self.audio_delay_token_id = tokenizer.convert_tokens_to_ids(self.audio_delay_token)
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id

        super().__init__(
            feature_extractor,
            tokenizer,
            audio_tokenizer=audio_tokenizer,
            chat_template=chat_template,
        )

    def get_audio_tokens(self, num_audio_tokens):
        pass

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[HiggsAudioV2ProcessorKwargs],
    ):
        output_kwargs = self._merge_kwargs(
            HiggsAudioV2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        n_audio_in_text = [t.count(self.audio_token) for t in text]

        n_audio = 0
        if audio is not None:
            audio = make_list_of_audio(audio)
            n_audio = len(audio)

        if sum(n_audio_in_text) > 0 and n_audio != sum(n_audio_in_text):
            if audio is None:
                raise ValueError("No audio were provided, but there are audio tokens in the prompt")
            else:
                raise ValueError(
                    f"The number of audio tokens in each text ({n_audio_in_text}) should be the same as the "
                    f"number of provided audios ({n_audio})."
                )
        elif sum(n_audio_in_text) == 0 and n_audio > 0:
            raise ValueError("Audio were provided, but there are no audio tokens in the prompt")

        if audio is not None:
            audio_input_ids_list = []
            for audio_el in audio:
                audio_inputs = self.feature_extractor(audio_el, **audio_kwargs)

                audio_inputs.pop("padding_mask", None)
                audio_inputs.to(self.audio_tokenizer.device)
                audio_input_ids = self.audio_tokenizer.encode(**audio_inputs).audio_codes

                bos_codes = audio_input_ids.new_full((*audio_input_ids.shape[:2], 1), self.audio_stream_bos_id)
                eos_codes = audio_input_ids.new_full((*audio_input_ids.shape[:2], 1), self.audio_stream_eos_id)
                audio_input_ids = torch.cat([bos_codes, audio_input_ids, eos_codes], dim=2)

                audio_input_ids = self.build_delay_pattern(audio_input_ids)
                audio_input_ids_list.append(audio_input_ids[0].transpose(0, 1))

            num_audio_tokens_iter = iter(len(audio_input_ids) for audio_input_ids in audio_input_ids_list)
            for i in range(len(text)):
                expanded = re.sub(
                    re.escape(self.audio_token), lambda _: self.get_audio_tokens(next(num_audio_tokens_iter)), text[i]
                )
                text[i] = expanded

            audio_input_ids_iter = iter(audio_input_ids_list)
            audio_input_ids_list = [list(islice(audio_input_ids_iter, l)) for l in n_audio_in_text]
            audio_input_ids_list = [torch.cat(batch_el, dim=0) for batch_el in audio_input_ids_list]

            lengths = [ids.shape[0] for ids in audio_input_ids_list]
            max_length = max(lengths)
            audio_input_ids_list = [
                F.pad(ids, (0, 0, 0, max_length - ids.shape[0]), value=self.audio_stream_eos_id)
                for ids in audio_input_ids_list
            ]
            audio_input_ids = torch.stack(audio_input_ids_list, dim=0)
            audio_input_ids_mask = torch.arange(max_length)[None, :] < torch.tensor(lengths)[:, None]

        data = self.tokenizer(text, **text_kwargs)
        if audio is not None:
            data.update(
                {
                    "audio_input_ids": audio_input_ids,
                    "audio_input_ids_mask": audio_input_ids_mask,
                }
            )

        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels[labels == self.audio_bos_token_id] = -100
            data["labels"] = labels

            if audio is not None:
                audio_labels = audio_input_ids.clone()
                audio_labels[audio_labels == self.audio_stream_bos_id] = -100
                audio_labels[audio_labels == self.audio_stream_eos_id] = -100
                data.update({"audio_labels": audio_labels})

        return BatchFeature(data=data, tensor_type="pt")

    def batch_decode(self, audio_input_ids):
        """
        Decode a batch of audio token sequences into audio waveforms.

        This method processes audio token sequences generated by the model, extracting the actual audio tokens
        between the beginning-of-stream (BOS) and end-of-stream (EOS) markers, reverting the delay pattern
        used during generation, and decoding them into audio waveforms using the audio tokenizer.

        Args:
            audio_input_ids (`torch.LongTensor`):
                Shape `(batch_size, sequence_length, num_codebooks)`
                The audio token sequences to decode. These should contain audio tokens with BOS and EOS markers
                in a delay pattern format as generated by the model.

        Returns:
            `list[torch.Tensor]`: A list of decoded audio waveforms, one for each batch element. Each waveform
            is a 1D tensor containing the audio samples.
        """
        audio_bos_token_idxs = (audio_input_ids == self.audio_stream_bos_id).all(-1).nonzero()
        start_of_generation_idx = audio_bos_token_idxs[-1, -1].item()

        audio_input_ids = audio_input_ids[:, start_of_generation_idx:]

        audio_eos_token_idxs = (audio_input_ids == self.audio_stream_eos_id).all(-1).nonzero()
        end_of_generation_idxs = [
            audio_eos_token_idxs[audio_eos_token_idxs[:, 0] == batch_idx, 1].min().item()
            if len(audio_eos_token_idxs[audio_eos_token_idxs[:, 0] == batch_idx]) > 0
            else audio_input_ids.shape[1]
            for batch_idx in range(audio_input_ids.shape[0])
        ]

        audios = []
        with torch.no_grad():
            for batch_idx in range(audio_input_ids.shape[0]):
                audio_token_ids = audio_input_ids[batch_idx, 1 : end_of_generation_idxs[batch_idx]]
                audio_token_ids = self.revert_delay_pattern(audio_token_ids).clip(0, self.audio_stream_bos_id - 1)
                audio_i = (
                    self.audio_tokenizer.decode(audio_token_ids.transpose(0, 1).unsqueeze(0))
                    .audio_values.cpu()
                    .squeeze()
                )
                audios.append(audio_i)

        return audios

    def decode(self, audio_input_ids):
        if audio_input_ids.shape[0] != 1:
            raise ValueError(
                f"Expecting a single output to be decoded but received {audio_input_ids.shape[0]} samples instead."
            )

        return self.batch_decode(audio_input_ids)[0]

    def build_delay_pattern(self, input_ids):
        pass

    def revert_delay_pattern(self, input_ids):
        seq_len, num_codebooks = input_ids.shape
        slices = []
        for i in range(num_codebooks):
            end_idx = seq_len - num_codebooks + 1 + i
            slices.append(input_ids[i:end_idx, i : i + 1])

        return torch.cat(slices, dim=1)

    def save_audio(
        self,
        audio: AudioInput,
        saving_path: str | Path | list[str | Path],
        **kwargs: Unpack[HiggsAudioV2ProcessorKwargs],
    ):
        pass

    @property
    def model_input_names(self):
        pass


__all__ = ["HiggsAudioV2Processor"]
