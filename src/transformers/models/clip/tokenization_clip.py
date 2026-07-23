
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class CLIPTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        **kwargs,
    ):
        _vocab = (
            vocab
            if vocab is not None
            else {
                str(bos_token): 0,
                str(eos_token): 1,
                str(pad_token): 2,
            }
        )

        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=_vocab,
                merges=self._merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="</w>",
                fuse_unk=False,
                unk_token=str(unk_token),
            )
        )

        self._tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "), normalizers.Lowercase()]
        )

        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(
                        r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
                    ),
                    behavior="removed",
                    invert=True,
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )

        self._tokenizer.decoder = decoders.ByteLevel()

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

        self._tokenizer.post_processor = processors.RobertaProcessing(
            sep=(str(eos_token), self.eos_token_id),
            cls=(str(bos_token), self.bos_token_id),
            add_prefix_space=False,
            trim_offsets=False,
        )

        self._wrap_decode_method_backend_tokenizer()

    def _wrap_decode_method_backend_tokenizer(self):
        orig_decode_method = self.backend_tokenizer.decode

        end_of_word_suffix = self.backend_tokenizer.model.end_of_word_suffix

        def new_decode_method(*args, **kwargs):
            pass

        self.backend_tokenizer.decode = new_decode_method


__all__ = ["CLIPTokenizer"]
