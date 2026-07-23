
import copy
import re


class TrialShortNamer:
    PREFIX = "hp"
    DEFAULTS = {}
    NAMING_INFO = None

    @classmethod
    def set_defaults(cls, prefix, defaults):
        cls.PREFIX = prefix
        cls.DEFAULTS = defaults
        cls.build_naming_info()

    @staticmethod
    def shortname_for_word(info, word):
        if len(word) == 0:
            return ""
        short_word = None
        if any(char.isdigit() for char in word):
            raise Exception(f"Parameters should not contain numbers: '{word}' contains a number")
        if word in info["short_word"]:
            return info["short_word"][word]
        for prefix_len in range(1, len(word) + 1):
            prefix = word[:prefix_len]
            if prefix in info["reverse_short_word"]:
                continue
            else:
                short_word = prefix
                break

        if short_word is None:
            def int_to_alphabetic(integer):
                s = ""
                while integer != 0:
                    s = chr(ord("A") + integer % 10) + s
                    integer //= 10
                return s

            i = 0
            while True:
                sword = word + "#" + int_to_alphabetic(i)
                if sword in info["reverse_short_word"]:
                    continue
                else:
                    short_word = sword
                    break

        info["short_word"][word] = short_word
        info["reverse_short_word"][short_word] = word
        return short_word

    @staticmethod
    def shortname_for_key(info, param_name):
        words = param_name.split("_")

        shortname_parts = [TrialShortNamer.shortname_for_word(info, word) for word in words]

        separators = ["", "_"]

        for separator in separators:
            shortname = separator.join(shortname_parts)
            if shortname not in info["reverse_short_param"]:
                info["short_param"][param_name] = shortname
                info["reverse_short_param"][shortname] = param_name
                return shortname

        return param_name

    @staticmethod
    def add_new_param_name(info, param_name):
        short_name = TrialShortNamer.shortname_for_key(info, param_name)
        info["short_param"][param_name] = short_name
        info["reverse_short_param"][short_name] = param_name

    @classmethod
    def build_naming_info(cls):
        if cls.NAMING_INFO is not None:
            return

        info = {
            "short_word": {},
            "reverse_short_word": {},
            "short_param": {},
            "reverse_short_param": {},
        }

        field_keys = list(cls.DEFAULTS.keys())

        for k in field_keys:
            cls.add_new_param_name(info, k)

        cls.NAMING_INFO = info

    @classmethod
    def shortname(cls, params):
        pass

    @classmethod
    def parse_repr(cls, repr):
        pass
