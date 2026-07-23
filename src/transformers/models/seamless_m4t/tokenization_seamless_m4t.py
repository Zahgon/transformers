
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_python import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
)
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import PaddingStrategy, logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}


class SeamlessM4TTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    prefix_tokens: list[int] = None
    suffix_tokens: list[int] = None

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        src_lang="eng",
        tgt_lang="fra",
        additional_special_tokens=None,
        keep_accents=None,
        vocab_file=None,
        **kwargs,
    ):
        self._vocab = vocab or {
            str(pad_token): 0,
            str(unk_token): 1,
            str(bos_token): 2,
            str(eos_token): 3,
        }

        self._merges = merges or []
        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                unk_token=str(unk_token),
                fuse_unk=True,
                byte_fallback=False,
            )
        )

        self._tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Replace(Regex(r"[\n\r\t]"), " "),
                normalizers.NFKC(),
                normalizers.Strip(left=False, right=True),
                normalizers.Replace(Regex(r" +▁"), "▁"),
                normalizers.Replace(Regex(r"^▁+$"), ""),
                normalizers.Replace(Regex(r" {2,}"), "▁"),
            ]
        )

        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first", split=True)

        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="first", split=True)

        if "__" not in src_lang:
            src_lang = f"__{src_lang}__"
        if "__" not in tgt_lang:
            tgt_lang = f"__{tgt_lang}__"

        if additional_special_tokens is not None:
            kwargs.setdefault("additional_special_tokens", additional_special_tokens)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            keep_accents=keep_accents,
            vocab_file=vocab_file,
            **kwargs,
        )

        self.fairseq_offset = 1
        self.fairseq_tokens_to_ids = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
        }
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        self._src_lang = src_lang
        self._tgt_lang = tgt_lang

        self.set_tgt_lang_special_tokens(self._tgt_lang)

    @classmethod
    def convert_from_spm_model(cls, vocab, **kwargs):
        """When converting from spm, offset is needed to account for special tokens."""
        _vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
        }
        for i, token in enumerate(list(vocab.keys())):
            _vocab[token] = i + 1  # offset by 1 to account for special tokens
        kwargs["vocab"] = _vocab
        return kwargs

    @property
    def src_lang(self) -> str:
        pass

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        pass

    @property
    def tgt_lang(self) -> str:
        pass

    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang: str) -> None:
        pass

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: str | None, tgt_lang: str | None, **extra_kwargs
    ):
        pass

    def prepare_seq2seq_batch(
        self,
        src_texts: list[str],
        src_lang: str = "eng",
        tgt_texts: list[str] | None = None,
        tgt_lang: str = "fra",
        max_length: int | None = None,
        max_target_length: int | None = None,
        padding: str = "longest",
        return_tensors: str | None = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        pass

    def _switch_to_input_mode(self):
        pass

    def _switch_to_target_mode(self):
        pass

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting.
        Prefix=[src_lang_code], suffix = [eos]
        """
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)

        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`src_lang={src_lang}` has not be found in the `vocabulary`. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target lang setting.
        Prefix=[eos, tgt_lang_code] and suffix=[eos].
        """
        self.cur_lang_code = self.convert_tokens_to_ids(lang)

        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`tgt_lang={lang}` has not be found in the `vocabulary`. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        self.prefix_tokens = [self.eos_token_id, self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        padding: bool | str | PaddingStrategy = False,
        pad_to_multiple_of: int | None = None,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        **kwargs,
    ):
        """
        Args:
            text (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_target (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair_target (`str`, `list[str]`, `list[list[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            pad_to_multiple_of (`int`, *optional*, defaults to `None`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            src_lang (`str`, *optional*):
                A string representing the source language. If not specified, the last `src_lang` specified (either
                during initialization or when calling this tokenizer) will be used.
            tgt_lang (`str`, *optional*):
                A string representing the target language. If not specified, the last `tgt_lang` specified (either
                during initialization or when calling this tokenizer) will be used.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to [`TokenizersBackend.__call__`].
        """
        if src_lang is not None:
            self.src_lang = src_lang
        if tgt_lang is not None:
            self.tgt_lang = tgt_lang

        output = super().__call__(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            **kwargs,
        )

        return output


__all__ = ["SeamlessM4TTokenizer"]
