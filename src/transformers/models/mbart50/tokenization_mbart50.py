

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_python import AddedToken, BatchEncoding
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}


FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]  # fmt: skip


class MBart50Tokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    prefix_tokens: list[int] = []
    suffix_tokens: list[int] = []

    def __init__(
        self,
        vocab: str | dict | list | None = None,
        _spm_precompiled_charsmap: str | None = None,
        src_lang=None,
        tgt_lang=None,
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs,
    ):
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token


        if isinstance(vocab, list):

            vocab = [(str(item[0]), float(item[1])) for item in vocab]

            vocab_tokens = [item[0] for item in vocab]
            has_language_codes = any(lang_code in vocab_tokens for lang_code in FAIRSEQ_LANGUAGE_CODES)

            if has_language_codes:
                self._vocab_scores = vocab
            else:

                vocab_list = [
                    (str(cls_token), 0.0),  # 0: <s>
                    (str(pad_token), 0.0),  # 1: <pad>
                    (str(eos_token), 0.0),  # 2: </s>
                    (str(unk_token), 0.0),  # 3: <unk>
                ]
                vocab_list.extend(vocab[3:])

                for lang_code in FAIRSEQ_LANGUAGE_CODES:
                    vocab_list.append((str(lang_code), 0.0))

                vocab_list.append((str(mask_token), 0.0))

                self._vocab_scores = vocab_list
        else:
            self._vocab_scores = [
                (str(cls_token), 0.0),
                (str(pad_token), 0.0),
                (str(eos_token), 0.0),
                (str(unk_token), 0.0),
                ("▁", -2.0),
            ]
            for lang_code in FAIRSEQ_LANGUAGE_CODES:
                self._vocab_scores.append((lang_code, 0.0))
            self._vocab_scores.append((str(mask_token), 0.0))

        self._tokenizer = Tokenizer(
            Unigram(
                self._vocab_scores,
                unk_id=3,
                byte_fallback=False,
            )
        )

        normalizers_ = [normalizers.Replace(Regex(r" {2,}"), " ")]
        if _spm_precompiled_charsmap is not None:
            normalizers_ = [normalizers.Precompiled(_spm_precompiled_charsmap)] + normalizers_

        self._tokenizer.normalizer = normalizers.Sequence(normalizers_)
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)

        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)
        additional_special_tokens = kwargs.pop("additional_special_tokens", []) or []
        additional_special_tokens.extend(FAIRSEQ_LANGUAGE_CODES)
        super().__init__(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.fairseq_offset = 1

        try:
            lang_tokens = [AddedToken(code, special=True) for code in FAIRSEQ_LANGUAGE_CODES]
        except Exception:
            lang_tokens = list(FAIRSEQ_LANGUAGE_CODES)
        existing_extra = getattr(self, "_extra_special_tokens", []) or []
        existing_strs = {str(t) for t in existing_extra}
        merged_extra = list(existing_extra) + [t for t in lang_tokens if str(t) not in existing_strs]
        self._extra_special_tokens = merged_extra

        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.tgt_lang = tgt_lang

        self._build_language_code_mappings()

        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.set_src_lang_special_tokens(self._src_lang)

    def _build_language_code_mappings(self):
        """Build language code to ID mappings and fairseq compatibility mappings."""
        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}

        self.fairseq_tokens_to_ids = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3,
        }
        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        mask_token = getattr(self, "mask_token", "<mask>")
        self.fairseq_tokens_to_ids["<mask>"] = self.convert_tokens_to_ids(str(mask_token))
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

    def _post_init(self):
        """Called after tokenizer.json is loaded in from_pretrained."""
        self._build_language_code_mappings()
        if hasattr(self, "_src_lang"):
            self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
            self.set_src_lang_special_tokens(self._src_lang)

    @property
    def src_lang(self) -> str:
        pass

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        pass

    def prepare_seq2seq_batch(
        self,
        src_texts: list[str],
        src_lang: str = "en_XX",
        tgt_texts: list[str] | None = None,
        tgt_lang: str = "ro_RO",
        **kwargs,
    ) -> BatchEncoding:
        pass

    def _switch_to_input_mode(self):
        pass

    def _switch_to_target_mode(self):
        pass

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos]."""
        self.cur_lang_code_id = self.convert_tokens_to_ids(src_lang)
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[tgt_lang_code] and suffix=[eos]."""
        self.cur_lang_code_id = self.convert_tokens_to_ids(tgt_lang)
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: str | None, tgt_lang: str | None, **extra_kwargs
    ):
        pass


__all__ = ["MBart50Tokenizer"]

MBart50TokenizerFast = MBart50Tokenizer
