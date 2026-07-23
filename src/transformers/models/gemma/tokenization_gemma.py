
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}


class GemmaTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        if vocab is None:
            vocab = {
                str(pad_token): 0,
                str(eos_token): 1,
                str(bos_token): 2,
                str(unk_token): 3,
                str(mask_token): 4,
            }
        self._vocab = vocab
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                fuse_unk=True,
                unk_token=str(unk_token),
                dropout=None,
                byte_fallback=True,
            )
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.Split(
            pattern=" ", behavior="merged_with_previous", invert=False
        )

        self._tokenizer.decoder = decoders.Sequence(
            [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
        )
        self._tokenizer.normalizer = normalizers.Replace(" ", "▁")
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )


__all__ = ["GemmaTokenizer"]
