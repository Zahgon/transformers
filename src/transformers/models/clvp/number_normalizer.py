

import sys


if sys.version_info >= (3, 11):
    import re
else:
    import regex as re


class EnglishNormalizer:
    def __init__(self):
        self._abbreviations = [
            (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
            for x in [
                ("mrs", "misess"),
                ("mr", "mister"),
                ("dr", "doctor"),
                ("st", "saint"),
                ("co", "company"),
                ("jr", "junior"),
                ("maj", "major"),
                ("gen", "general"),
                ("drs", "doctors"),
                ("rev", "reverend"),
                ("lt", "lieutenant"),
                ("hon", "honorable"),
                ("sgt", "sergeant"),
                ("capt", "captain"),
                ("esq", "esquire"),
                ("ltd", "limited"),
                ("col", "colonel"),
                ("ft", "fort"),
            ]
        ]

        self.ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.teens = [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        self.tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def number_to_words(self, num: int) -> str:
        pass

    def convert_to_ascii(self, text: str) -> str:
        pass

    def _expand_dollars(self, m: str) -> str:
        pass

    def _remove_commas(self, m: str) -> str:
        pass

    def _expand_decimal_point(self, m: str) -> str:
        pass

    def _expand_ordinal(self, num: str) -> str:
        pass

    def _expand_number(self, m: str) -> str:
        pass

    def normalize_numbers(self, text: str) -> str:
        pass

    def expand_abbreviations(self, text: str) -> str:
        pass

    def collapse_whitespace(self, text: str) -> str:
        pass

    def __call__(self, text):
        """
        Converts text to ascii, numbers / number-like quantities to their spelt-out counterparts and expands
        abbreviations
        """

        text = self.convert_to_ascii(text)
        text = text.lower()
        text = self.normalize_numbers(text)
        text = self.expand_abbreviations(text)
        text = self.collapse_whitespace(text)
        text = text.replace('"', "")

        return text
