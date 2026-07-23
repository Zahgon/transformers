
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from ...utils import auto_docstring


class MoonshineStreamingProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "pad_to_multiple_of": 80,
            "padding": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class MoonshineStreamingProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput | None = None,
        text: str | list[str] | TextInput | PreTokenizedInput | None = None,
        **kwargs: Unpack[MoonshineStreamingProcessorKwargs],
    ):
        r"""
        Returns:
            This method returns the results of each `call` method. If both are used, the output is a dictionary containing the results of both.
        """
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        output_kwargs = self._merge_kwargs(
            MoonshineStreamingProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def pad(self, *args, **kwargs):
        """
        This method operates on batches of extracted features and/or tokenized text. It forwards all arguments to
        [`MoonshineStreamingFeatureExtractor.pad`] and/or [`PreTrainedTokenizer.pad`] depending on the input modality and returns their outputs. If both modalities are passed, [`MoonshineStreamingFeatureExtractor.pad`] and [`PreTrainedTokenizer.pad`] are called.

        Args:
            input_features:
                When the first argument is a dictionary containing a batch of tensors, or the `input_features` argument is present, it is passed to [`MoonshineStreamingFeatureExtractor.pad`].
            labels:
                When the `label` argument is present, it is passed to [`PreTrainedTokenizer.pad`].

        Returns:
            This method returns the results of each `pad` method. If both are used, the output is a dictionary containing the results of both.
        """
        input_features = kwargs.pop("input_features", None)
        labels = kwargs.pop("labels", None)
        if len(args) > 0:
            input_features = args[0]
            args = args[1:]

        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            input_features["labels"] = labels["input_ids"]
            return input_features

    @property
    def model_input_names(self):
        pass


__all__ = ["MoonshineStreamingProcessor"]
