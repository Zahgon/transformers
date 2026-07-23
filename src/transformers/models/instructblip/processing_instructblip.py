
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class InstructBlipProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
        },
    }


@auto_docstring
class InstructBlipProcessor(ProcessorMixin):
    def __init__(self, image_processor, tokenizer, qformer_tokenizer, num_query_tokens=None, **kwargs):
        r"""
        qformer_tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The Q-Former tokenizer is a required input.
        num_query_tokens (`int`, *optional*):
            "
            Number of tokens used by the Qformer as queries, should be same as in model's config.
        """
        if not hasattr(tokenizer, "image_token"):
            self.image_token = AddedToken("<image>", normalized=False, special=True)
            tokenizer.add_tokens([self.image_token], special_tokens=True)
        else:
            self.image_token = tokenizer.image_token
        self.num_query_tokens = num_query_tokens

        super().__init__(image_processor, tokenizer, qformer_tokenizer)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[InstructBlipProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You have to specify at least images or text.")

        output_kwargs = self._merge_kwargs(
            InstructBlipProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        encoding = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            qformer_text_encoding = self.qformer_tokenizer(text, **output_kwargs["text_kwargs"])
            encoding["qformer_input_ids"] = qformer_text_encoding.pop("input_ids")
            encoding["qformer_attention_mask"] = qformer_text_encoding.pop("attention_mask")

            if output_kwargs["text_kwargs"].get("max_length") is not None:
                output_kwargs["text_kwargs"]["max_length"] -= self.num_query_tokens
            text_encoding = self.tokenizer(text, **output_kwargs["text_kwargs"])

            if images is not None:
                image_tokens = self.image_token.content * self.num_query_tokens
                output_kwargs["text_kwargs"]["add_special_tokens"] = False
                output_kwargs["text_kwargs"]["padding"] = False
                output_kwargs["text_kwargs"]["truncation"] = False
                image_text_encoding = self.tokenizer(image_tokens, **output_kwargs["text_kwargs"])
                for k in text_encoding:
                    text_encoding[k] = [image_text_encoding[k] + sample for sample in text_encoding[k]]
            encoding.update(text_encoding)

        if images is not None:
            image_encoding = self.image_processor(images, **output_kwargs["images_kwargs"])
            encoding.update(image_encoding)

        encoding = BatchFeature(encoding, tensor_type=return_tensors)
        return encoding

    @property
    def model_input_names(self):
        pass


__all__ = ["InstructBlipProcessor"]
