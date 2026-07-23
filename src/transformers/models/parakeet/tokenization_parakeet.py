
import itertools

from ...tokenization_utils_tokenizers import TokenizersBackend


class ParakeetTokenizer(TokenizersBackend):

    def _decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        group_tokens: bool = True,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if group_tokens:
            token_ids = [token_group[0] for token_group in itertools.groupby(token_ids)]

        token_ids = [token for token in token_ids if token != self.pad_token_id]

        return super()._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["ParakeetTokenizer"]
