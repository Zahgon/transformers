
import collections

from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, add_end_docstrings, add_start_docstrings, logging
from ..bert.tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


class DPRContextEncoderTokenizer(BertTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(self, *args, do_lower_case=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_lower_case = do_lower_case


class DPRQuestionEncoderTokenizer(BertTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(self, *args, do_lower_case=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_lower_case = do_lower_case


DPRSpanPrediction = collections.namedtuple(
    "DPRSpanPrediction", ["span_score", "relevance_score", "doc_id", "start_index", "end_index", "text"]
)

DPRReaderOutput = collections.namedtuple("DPRReaderOutput", ["start_logits", "end_logits", "relevance_logits"])


CUSTOM_DPR_READER_DOCSTRING = r"""
    Return a dictionary with the token ids of the input strings and other information to give to `.decode_best_spans`.
    It converts the strings of a question and different passages (title and text) in a sequence of IDs (integers),
    using the tokenizer and vocabulary. The resulting `input_ids` is a matrix of size `(n_passages, sequence_length)`
    with the format:

    ```
    [CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>
    ```

    Args:
        questions (`str` or `list[str]`):
            The questions to be encoded. You can specify one question for many passages. In this case, the question
            will be duplicated like `[questions] * n_passages`. Otherwise you have to specify as many questions as in
            `titles` or `texts`.
        titles (`str` or `list[str]`):
            The passages titles to be encoded. This can be a string or a list of strings if there are several passages.
        texts (`str` or `list[str]`):
            The passages texts to be encoded. This can be a string or a list of strings if there are several passages.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
            Activates and controls padding. Accepts the following values:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
            Activates and controls truncation. Accepts the following values:

            - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to
              the maximum acceptable input length for the model if that argument is not provided. This will truncate
              token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch
              of pairs) is provided.
            - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided. This will only truncate the first
              sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided. This will only truncate the
              second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
              greater than the model maximum admissible input size).
        max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        return_attention_mask (`bool`, *optional*):
            Whether or not to return the attention mask. If not set, will return the attention mask according to the
            specific tokenizer's default, defined by the `return_outputs` attribute.

            [What are attention masks?](../glossary#attention-mask)

    Returns:
        `dict[str, list[list[int]]]`: A dictionary with the following keys:

        - `input_ids`: List of token ids to be fed to a model.
        - `attention_mask`: List of indices specifying which tokens should be attended to by the model.
    """


@add_start_docstrings(CUSTOM_DPR_READER_DOCSTRING)
class CustomDPRReaderTokenizerMixin:
    def __call__(
        self,
        questions,
        titles: str | None = None,
        texts: str | None = None,
        padding: bool | str = False,
        truncation: bool | str = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        **kwargs,
    ) -> BatchEncoding:
        if titles is None and texts is None:
            return super().__call__(
                questions,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )
        elif titles is None or texts is None:
            text_pair = titles if texts is None else texts
            return super().__call__(
                questions,
                text_pair,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )
        titles = titles if not isinstance(titles, str) else [titles]
        texts = texts if not isinstance(texts, str) else [texts]
        n_passages = len(titles)
        questions = questions if not isinstance(questions, str) else [questions] * n_passages
        if len(titles) != len(texts):
            raise ValueError(
                f"There should be as many titles than texts but got {len(titles)} titles and {len(texts)} texts."
            )
        encoded_question_and_titles = super().__call__(questions, titles, padding=False, truncation=False)["input_ids"]
        encoded_texts = super().__call__(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        encoded_inputs = {
            "input_ids": [
                (encoded_question_and_title + encoded_text)[:max_length]
                if max_length is not None and truncation
                else encoded_question_and_title + encoded_text
                for encoded_question_and_title, encoded_text in zip(encoded_question_and_titles, encoded_texts)
            ]
        }
        if return_attention_mask is not False:
            attention_mask = []
            for input_ids in encoded_inputs["input_ids"]:
                attention_mask.append([int(input_id != self.pad_token_id) for input_id in input_ids])
            encoded_inputs["attention_mask"] = attention_mask
        return self.pad(encoded_inputs, padding=padding, max_length=max_length, return_tensors=return_tensors)

    def decode_best_spans(
        self,
        reader_input: BatchEncoding,
        reader_output: DPRReaderOutput,
        num_spans: int = 16,
        max_answer_length: int = 64,
        num_spans_per_passage: int = 4,
    ) -> list[DPRSpanPrediction]:
        pass

    def _get_best_spans(
        self,
        start_logits: list[int],
        end_logits: list[int],
        max_answer_length: int,
        top_spans: int,
    ) -> list[DPRSpanPrediction]:
        pass


@add_end_docstrings(CUSTOM_DPR_READER_DOCSTRING)
class DPRReaderTokenizer(CustomDPRReaderTokenizerMixin, BertTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, *args, do_lower_case=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_lower_case = do_lower_case


__all__ = ["DPRContextEncoderTokenizer", "DPRQuestionEncoderTokenizer", "DPRReaderOutput", "DPRReaderTokenizer"]
