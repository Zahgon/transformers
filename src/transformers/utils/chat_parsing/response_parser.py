
from __future__ import annotations

from typing import Any

from .content_parsers import STREAMABLE_PARSERS, process_field
from .response_templates import ResponseTemplate, ResponseTemplateField, load_response_template


def parse_response(text: str, response_template: dict | ResponseTemplate, *, prefix: str | None = None) -> dict:
    """The main function for response parsing when you don't want streaming. Takes generated output
    and the prompt prefix and parses them without streaming any events, then returns the parsed message.
    """
    response_template = load_response_template(response_template)
    stream = ResponseParser(response_template, prefix=prefix)
    stream.feed(text)
    message, _ = stream.finalize()
    return message


class ResponseParser:

    def __init__(self, response_template: dict | ResponseTemplate, prefix: str | None = None):
        self._spec = load_response_template(response_template)
        if prefix is None:
            raise ValueError(
                "`ResponseParser`/`parse_response` requires `prefix` (the chat prompt sent to the model before "
                "generation), because chat templates often pre-write part of the assistant message (e.g. an "
                "opening `<think>` tag) that the parser must see to parse the output correctly. If the generation "
                'already contains the complete message, pass `prefix=""` to opt out explicitly.'
            )
        self._buffer: str = ""
        self._pos: int = 0
        self._output: dict[str, Any] = dict(self._spec.defaults)
        self._implicit_name: str | None = self._spec.implicit
        self._current: str | None = self._implicit_name
        self._captures: dict[str, str] = {}
        self._body: str = ""
        self._opened: bool = False
        self._finalized: bool = False
        self.initial_events: list[dict] = []
        if prefix:
            self._consume_prefix(prefix)

    def _consume_prefix(self, prefix: str) -> None:
        """Loads the prefix (the chat prefill sent to the model), right-truncates it to the start of the
        assistant message (as determined by start_anchor) and then runs the remainder through the parser.
        Events produced while processing the prefix are stashed on `initial_events` so callers can replay
        them into a renderer before feeding model output.

        Think of this as the "get the parser up to speed on the story so far" method.
        """
        truncated = self._spec.truncate_past_last_anchor(prefix)
        if not truncated:
            return
        self._buffer = truncated
        self._process(self.initial_events, eos=False)

    def feed(self, text: str) -> list[dict]:
        """Feeds more text/tokens from the model output into the tokenizer, and returns any events that result
        (regions entered or left). This is the method you want to call after each generation step."""
        if self._finalized:
            raise RuntimeError("ResponseParser already finalized")
        if text:
            self._buffer += text
        events: list[dict] = []
        self._process(events, eos=False)
        return events

    def finalize(self) -> tuple[dict, list[dict]]:
        """Close the stream and return the final message dict together with
        any finalization events. This is necessary because some regions may only
        end at the end of the sequence, so you won't see the event telling you they're
        ready until the sequence is finalized."""

        def _is_empty(v: Any) -> bool:
            return v is None or (isinstance(v, (list, dict, str)) and not v)

        if self._finalized:
            raise RuntimeError("ResponseParser already finalized")
        events: list[dict] = []
        self._process(events, eos=True)
        missing = [n for n, f in self._spec.fields.items() if not f.optional and n not in self._output]
        if missing:
            raise ValueError(f"Required response_template fields missing from parsed output: {missing}")
        defaults = self._spec.defaults
        self._output = {k: v for k, v in self._output.items() if k in defaults or not _is_empty(v)}
        self._finalized = True
        return self._output, events

    def _process(self, events: list[dict], eos: bool) -> None:
        while True:
            watch = self._watchlist()
            best, hold_start = self._scan(watch, eos)

            if best is not None:
                kind, field, m = best
                if m.start() > self._pos:
                    self._accumulate(events, self._buffer[self._pos : m.start()])
                self._pos = m.end()
                if kind == "open":
                    self._close_current(events)
                    self._open_explicit(events, field, m)
                else:  # "close" (always the implicit region's close here,
                    had_content = self._opened
                    self._close_current(events)
                    if not had_content and m.start() == m.end():
                        break
                continue

            if eos:
                if self._pos < len(self._buffer):
                    self._accumulate(events, self._buffer[self._pos :])
                    self._pos = len(self._buffer)
                self._close_current(events)
                break
            if hold_start > self._pos:
                self._accumulate(events, self._buffer[self._pos : hold_start])
                self._pos = hold_start
            break

    def _watchlist(self) -> list[tuple[str, ResponseTemplateField]]:
        """Patterns we care about right now: the close of the currently-open
        explicit region, or -- if we're in the implicit/null region -- every
        explicit open plus the implicit's own close (if any)."""
        if self._current is not None and self._current != self._implicit_name:
            field = self._spec.fields[self._current]
            return [("close", field)] if field.close_re is not None else []
        watch: list[tuple[str, ResponseTemplateField]] = []
        for field in self._spec.fields.values():
            if field.open_re is not None:
                watch.append(("open", field))
        if self._implicit_name is not None:
            impl = self._spec.fields[self._implicit_name]
            if impl.close_re is not None:
                watch.append(("close", impl))
        return watch

    def _scan(
        self, watch: list[tuple[str, ResponseTemplateField]], eos: bool
    ) -> tuple[tuple[str, ResponseTemplateField, Any] | None, int]:
        """Single pass over the watched delimiters, using the `regex` module's
        partial matching to decide -- per delimiter -- whether it can be committed
        now or must be held. Returns `(best, hold_start)`:

        * `best` is the earliest-starting delimiter we can safely commit *now*
          (longest on ties, opens before closes), or `None`.
        * `hold_start` is the leftmost buffer position occupied by a still-pending
          match: a partial (incomplete) delimiter, or a complete one ending at the
          buffer edge that more input could still grow. Bytes before it are safe to
          emit; bytes from it onward must be held. It stays `len(self._buffer)` when
          nothing is pending, letting the caller flush the whole buffer.

        A complete match is committable only if it starts strictly before
        `hold_start` -- otherwise an earlier (or co-located) pending delimiter could
        turn out to be the real one. At EOS nothing can grow, so partial matching is
        skipped and every complete match is committable.

        (The `regex` module always reports the empty string as a live prefix, so a
        partial search with no real match returns a zero-width match at the buffer
        end; that lands in the pending branch with `start == len(self._buffer)`, a
        no-op for `hold_start`.)
        """
        best_key: tuple | None = None
        best: tuple[str, ResponseTemplateField, Any] | None = None
        hold_start = len(self._buffer)
        for kind, field in watch:
            pattern = field.open_re if kind == "open" else field.close_re
            if eos:
                m = pattern.search(self._buffer, self._pos)
            else:
                m = pattern.search(self._buffer, self._pos, partial=True)
            if m is None:
                continue
            if not eos and (m.partial or self._can_grow(kind, field, m)):
                hold_start = min(hold_start, m.start())
                continue
            key = (m.start(), -(m.end() - m.start()), 0 if kind == "open" else 1, field.name)
            if best_key is None or key < best_key:
                best_key, best = key, (kind, field, m)
        if best is not None and best[2].start() >= hold_start:
            best = None
        return best, hold_start

    def _can_grow(self, kind: str, field: ResponseTemplateField, m: Any) -> bool:
        r"""Whether a *complete* match ending at the current buffer edge could still
        change as more input arrives -- in which case we defer rather than commit. A
        match ending before the edge has already seen its terminating byte and is
        final. At the edge: zero-width matches (`$` / `\Z`) are only real at true
        EOS; a fully-present literal that no other literal in its set extends cannot
        grow (the fast path that keeps literal delimiters zero-latency); anything
        else (regex delimiters, prefix-overlapping literal lists) might."""
        if m.end() != len(self._buffer):
            return False
        if m.start() == m.end():
            return True
        literals, can_extend = (
            (field.open_literals, field.open_literal_can_extend)
            if kind == "open"
            else (field.close_literals, field.close_literal_can_extend)
        )
        return literals is None or can_extend

    def _accumulate(self, events: list[dict], text: str) -> None:
        """Route `text` into the currently active region. When the current
        region is the null sink (no implicit declared, no explicit open), we
        silently discard. Every routed chunk emits a `region_chunk` event so
        consumers can render live; `dirty=True` flags chunks from structured
        parsers (json, xml-inline, kv-lines) whose raw bytes will only be
        parsed into the final value on close."""
        if not text or self._current is None:
            return
        field = self._spec.fields[self._current]
        if not self._opened:
            events.append({"type": "region_open", "field": self._current})
            self._opened = True
        self._body += text
        dirty = field.content not in STREAMABLE_PARSERS
        events.append({"type": "region_chunk", "field": self._current, "text": text, "dirty": dirty})

    def _open_explicit(self, events: list[dict], field: ResponseTemplateField, m: Any) -> None:
        self._current = field.name
        self._captures = {k: v for k, v in m.groupdict().items() if v is not None}
        self._body = ""
        self._opened = True
        events.append({"type": "region_open", "field": field.name})

    def _close_current(self, events: list[dict]) -> None:
        """Close the current region and reset to the implicit/null region.
        Skipped (aside from the reset) when the current region never opened --
        avoids vacuous open/close pairs at every explicit boundary."""
        if self._current is None or not self._opened:
            self._reset_to_implicit()
            return
        field = self._spec.fields[self._current]
        value = process_field(self._body, field, self._captures)
        if field.repeats:
            self._output.setdefault(self._current, []).append(value)
        else:
            self._output[self._current] = value
        events.append({"type": "region_close", "field": self._current, "value": value})
        self._reset_to_implicit()

    def _reset_to_implicit(self) -> None:
        self._current = self._implicit_name
        self._captures = {}
        self._body = ""
        self._opened = False
