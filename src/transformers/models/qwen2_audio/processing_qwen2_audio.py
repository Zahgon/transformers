
import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class Qwen2AudioProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "audio_kwargs": {},
    }


@auto_docstring
class Qwen2AudioProcessor(ProcessorMixin):
    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|AUDIO|>",
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|AUDIO|>"`):
            The token to use for audio tokens.
        audio_bos_token (`str`, *optional*, defaults to `"<|audio_bos|>"`):
            The token to use for audio bos tokens.
        audio_eos_token (`str`, *optional*, defaults to `"<|audio_eos|>"`):
            The token to use for audio eos tokens.
        """
        if chat_template is None:
            chat_template = self.default_chat_template
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token = tokenizer.audio_bos_token if hasattr(tokenizer, "audio_bos_token") else audio_bos_token
        self.audio_eos_token = tokenizer.audio_eos_token if hasattr(tokenizer, "audio_eos_token") else audio_eos_token
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        audio: np.ndarray | list[np.ndarray] = None,
        **kwargs: Unpack[Qwen2AudioProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        output_kwargs = self._merge_kwargs(
            Qwen2AudioProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
            num_audios = 1 if type(audio) is np.ndarray else len(audio)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token{'s' if num_audio_tokens > 1 else ''} in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}"
                )

            output_kwargs["audio_kwargs"]["return_attention_mask"] = True
            output_kwargs["audio_kwargs"]["padding"] = "max_length"
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

            expanded_text = []
            audio_lengths = audio_inputs["feature_attention_mask"].sum(-1).tolist()

            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    audio_length = audio_lengths.pop(0)
                    input_length = (audio_length - 1) // 2 + 1
                    num_audio_tokens = (input_length - 2) // 2 + 1

                    expanded_audio_token = self.audio_token * num_audio_tokens

                    audio_token_start_idx = sample.find(self.audio_token)
                    audio_token_end_idx = audio_token_start_idx + len(self.audio_token)

                    has_bos = (
                        sample[audio_token_start_idx - len(self.audio_bos_token) : audio_token_start_idx]
                        == self.audio_bos_token
                    )
                    has_eos = (
                        sample[audio_token_end_idx : audio_token_end_idx + len(self.audio_eos_token)]
                        == self.audio_eos_token
                    )

                    if not has_bos and not has_eos:
                        expanded_audio_token = self.audio_bos_token + expanded_audio_token + self.audio_eos_token

                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, inputs, modalities=["audio"])

        if audio is not None:
            inputs.update(audio_inputs)

        return BatchFeature(data={**inputs}, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        pass

    @property
    def default_chat_template(self):
        pass


__all__ = ["Qwen2AudioProcessor"]
