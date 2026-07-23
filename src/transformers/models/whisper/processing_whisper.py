
from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring


@auto_docstring
class WhisperProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        pass

    @auto_docstring
    def __call__(self, *args, **kwargs):
        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs

        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def get_prompt_ids(self, text: str, return_tensors="np"):
        pass


__all__ = ["WhisperProcessor"]
