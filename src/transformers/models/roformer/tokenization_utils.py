
from tokenizers import NormalizedString, PreTokenizedString, normalizers


class JiebaPreTokenizer:
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self.normalizers = normalizers.BertNormalizer(
            clean_text=False,
            handle_chinese_chars=True,
            strip_accents=False,
            lowercase=False,
        )
        try:
            import rjieba
        except ImportError:
            raise ImportError(
                "You need to install rjieba to use RoFormerTokenizer. "
                "See https://pypi.org/project/rjieba/ for installation."
            )
        self.jieba = rjieba

    def jieba_split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
        pass

    def pre_tokenize(self, pretok: PreTokenizedString):
        pass
