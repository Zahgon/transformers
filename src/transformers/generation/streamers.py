
from __future__ import annotations

import asyncio
import sys
import time
from queue import Queue
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from ..tokenization_utils_base import PreTrainedTokenizerBase


class BaseStreamer:

    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        raise NotImplementedError()

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        raise NotImplementedError()


class TextStreamer(BaseStreamer):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, skip_prompt: bool = False, **decode_kwargs: Any):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        self.token_cache: list[int] = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.token_cache.extend(value.tolist())
        text = cast(str, self.tokenizer.decode(self.token_cache, **self.decode_kwargs))

        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        if len(self.token_cache) > 0:
            text = cast(str, self.tokenizer.decode(self.token_cache, **self.decode_kwargs))
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        print(text, flush=True, end="" if not stream_end else None)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True

        return False


class TextIteratorStreamer(TextStreamer):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        skip_prompt: bool = False,
        timeout: float | None = None,
        **decode_kwargs: Any,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class AsyncTextIteratorStreamer(TextStreamer):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        skip_prompt: bool = False,
        timeout: float | None = None,
        **decode_kwargs: Any,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = asyncio.Queue()
        self.stop_signal = None
        self.timeout = timeout
        self.loop = asyncio.get_running_loop()
        timeout_context = getattr(asyncio, "timeout", None)
        self.has_asyncio_timeout = sys.version_info >= (3, 11) and callable(timeout_context)
        self.asyncio_timeout = timeout_context if self.has_asyncio_timeout else None

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.loop.call_soon_threadsafe(self.text_queue.put_nowait, text)
        if stream_end:
            self.loop.call_soon_threadsafe(self.text_queue.put_nowait, self.stop_signal)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self.has_asyncio_timeout and self.asyncio_timeout is not None:
                async with self.asyncio_timeout(self.timeout):
                    value = await self.text_queue.get()
            else:
                value = await asyncio.wait_for(self.text_queue.get(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError()
        else:
            if value == self.stop_signal:
                raise StopAsyncIteration()
            else:
                return value


class TextDiffusionStreamer(TextStreamer):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        skip_prompt: bool = False,
        sleep_time: float | None = None,
        **decode_kwargs: Any,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self._has_draft = False
        self._takes_logits = False
        self.sleep_time = sleep_time

    def _clear_draft(self):
        if self._has_draft:
            print("\0338\033[J", end="", flush=True)
            self._has_draft = False

    def put_draft(self, value, **kwargs):
        """
        Receives the full sequence of draft tokens, decodes them, and prints them in yellow.
        Overwrites previous draft.
        """
        self._clear_draft()

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextDiffusionStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        text = self.tokenizer.decode(value, **self.decode_kwargs)

        print("\0337", end="", flush=True)
        print(f"\033[33m{text}\033[0m", end="", flush=True)
        self._has_draft = True
        if self.sleep_time is not None:
            time.sleep(self.sleep_time)

    def put(self, value):
        """Receives confirmed tokens, clears draft, and prints them permanently."""
        self._clear_draft()
        super().put(value)

    def end(self):
        """Flushes any remaining cache and prints a newline."""
        self._clear_draft()
        super().end()
