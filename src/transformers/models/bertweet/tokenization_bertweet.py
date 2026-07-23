
import html
import os
import re

import regex

from ...tokenization_python import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "merges_file": "bpe.codes",
}


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs


class BertweetTokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        merges_file,
        normalization=False,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs,
    ):
        try:
            from emoji import demojize

            self.demojizer = demojize
        except ImportError:
            logger.warning(
                "emoji is not installed, thus not converting emoticons or emojis into text. Install emoji: pip3"
                " install emoji==0.6.0"
            )
            self.demojizer = None

        self.vocab_file = vocab_file
        self.merges_file = merges_file

        self.encoder = {}
        self.encoder[str(bos_token)] = 0
        self.encoder[str(pad_token)] = 1
        self.encoder[str(eos_token)] = 2
        self.encoder[str(unk_token)] = 3

        self.add_from_file(vocab_file)

        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:-1]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

        self.normalization = normalization
        self.tweetPreprocessor = TweetTokenizer()
        self.special_puncts = {"’": "'", "…": "..."}

        super().__init__(
            normalization=normalization,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            token_type_ids_pattern="all_zeros",  # BERTweet doesn't use token type IDs
            token_type_ids_include_special_tokens=True,
            special_tokens_pattern="cls_double_sep",  # <s> X </s></s> Y </s>
            **kwargs,
        )

    @property
    def vocab_size(self):
        pass

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = "@@ ".join(word)
        word = word[:-4]
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        if self.normalization:  # Perform Tweet normalization before performing BPE
            text = self.normalizeTweet(text)

        split_tokens = []
        words = re.findall(r"\S+\n?", text)
        for token in words:
            split_tokens.extend(list(self.bpe(token).split(" ")))
        return split_tokens

    def normalizeTweet(self, tweet):
        """
        Normalize a raw Tweet
        """
        for punct in self.special_puncts:
            tweet = tweet.replace(punct, self.special_puncts[punct])

        tokens = self.tweetPreprocessor.tokenize(tweet)
        normTweet = " ".join([self.normalizeToken(token) for token in tokens])

        normTweet = (
            normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )

        return " ".join(normTweet.split())

    def normalizeToken(self, token):
        """
        Normalize tokens in a Tweet
        """
        lowercased_token = token.lower()
        if token.startswith("@"):
            return "@USER"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        elif len(token) == 1:
            if token in self.special_puncts:
                return self.special_puncts[token]
            if self.demojizer is not None:
                return self.demojizer(token)
            else:
                return token
        else:
            return token

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string


    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        """
        Save the vocabulary and merges files to a directory.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return ()

        vocab_files_names = getattr(self, "vocab_files_names", {})
        prefix = f"{filename_prefix}-" if filename_prefix else ""

        vocab_file = os.path.join(save_directory, prefix + vocab_files_names.get("vocab_file", "vocab.txt"))
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token, token_id in sorted(self.encoder.items(), key=lambda kv: kv[1]):
                if token_id >= 4:
                    f.write(f"{token} {token_id}\n")

        merge_file = os.path.join(save_directory, prefix + vocab_files_names.get("merges_file", "bpe.codes"))
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.writelines(
                " ".join(bpe_tokens) + "\n"
                for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1])
            )

        return (vocab_file, merge_file)

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(f"Incorrect encoding detected in {f}, please rebuild the dataset")
            return

        lines = f.readlines()
        for lineTmp in lines:
            line = lineTmp.strip()
            idx = line.rfind(" ")
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            self.encoder[word] = len(self.encoder)




"""
Twitter-aware tokenizer, designed to be flexible and easy to adapt to new domains and tasks. The basic logic is this:

1. The tuple regex_strings defines a list of regular expression strings.

2. The regex_strings strings are put, in order, into a compiled regular expression object called word_re.

3. The tokenization is done by word_re.findall(s), where s is the user-supplied string, inside the tokenize() method of
   the class Tokenizer.

4. When instantiating Tokenizer objects, there is a single option: preserve_case. By default, it is set to True. If it
   is set to False, then the tokenizer will lowercase everything except for emoticons.

"""




EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
    )"""

URLS = r"""			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |					#   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
    (?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"""

REGEXPS = (
    URLS,
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [ *\-.\)]*
      )?
      (?:            # (area code)
        [\(]?
        \d{3}
        [ *\-.\)]*
      )?
      \d{3}          # exchange
      [ *\-.\)]*
      \d{4}          # base
    )""",
    EMOTICONS,
    r"""<[^>\s]+>""",
    r"""[\-]+>|<[\-]+""",
    r"""(?:@[\w_]+)""",
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",
    r"""[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]""",
    r"""
    (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """,
)


WORD_RE = regex.compile(r"""(%s)""" % "|".join(REGEXPS), regex.VERBOSE | regex.I | regex.UNICODE)

HANG_RE = regex.compile(r"([^a-zA-Z0-9])\1{3,}")

EMOTICON_RE = regex.compile(EMOTICONS, regex.VERBOSE | regex.I | regex.UNICODE)

ENT_RE = regex.compile(r"&(#?(x?))([^&;\s]+);")




def _str_to_unicode(text, encoding=None, errors="strict"):
    if encoding is None:
        encoding = "utf-8"
    if isinstance(text, bytes):
        return text.decode(encoding, errors)
    return text


def _replace_html_entities(text, keep=(), remove_illegal=True, encoding="utf-8"):
    """
    Remove entities from text by converting them to their corresponding unicode character.

    Args:
        text:
            A unicode string or a byte string encoded in the given *encoding* (which defaults to 'utf-8').
        keep (list):
            List of entity names which should not be replaced. This supports both numeric entities (`&#nnnn;` and
            `&#hhhh;`) and named entities (such as `&nbsp;` or `&gt;`).
        remove_illegal (bool):
            If `True`, entities that can't be converted are removed. Otherwise, entities that can't be converted are
            kept "as is".
    Returns: A unicode string with the entities removed.

    See https://github.com/scrapy/w3lib/blob/master/w3lib/html.py

    Examples:

    ```python
    >>> from nltk.tokenize.casual import _replace_html_entities

    >>> _replace_html_entities(b"Price: &pound;100")
    'Price: \\xa3100'

    >>> print(_replace_html_entities(b"Price: &pound;100"))
    Price: £100
    ```"""

    def _convert_entity(match):
        pass

    return ENT_RE.sub(_convert_entity, _str_to_unicode(text, encoding))




class TweetTokenizer:

    def __init__(self, preserve_case=True, reduce_len=False, strip_handles=False):
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles

    def tokenize(self, text):
        """
        Args:
            text: str

        Returns: list(str) A tokenized list of strings; concatenating this list returns the original string if
        `preserve_case=False`
        """
        text = _replace_html_entities(text)
        if self.strip_handles:
            text = remove_handles(text)
        if self.reduce_len:
            text = reduce_lengthening(text)
        safe_text = HANG_RE.sub(r"\1\1\1", text)
        words = WORD_RE.findall(safe_text)
        if not self.preserve_case:
            words = [x if EMOTICON_RE.search(x) else x.lower() for x in words]
        return words




def reduce_lengthening(text):
    """
    Replace repeated character sequences of length 3 or greater with sequences of length 3.
    """
    pattern = regex.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1\1", text)


def remove_handles(text):
    """
    Remove Twitter username handles from text.
    """
    pattern = regex.compile(
        r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)"
    )
    return pattern.sub(" ", text)




def casual_tokenize(text, preserve_case=True, reduce_len=False, strip_handles=False):
    pass




__all__ = ["BertweetTokenizer"]
