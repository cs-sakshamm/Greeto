"""Streaming helpers for token-by-token LLM output.

The runtime prompt asks for one JSON object, so a streaming consumer cannot
hand raw tokens to TTS: the first tokens are ``{"reply": "`` and a sentence
may arrive in pieces. :class:`ReplyStreamExtractor` decodes the ``reply``
string incrementally (handling escapes and unicode), and
:class:`SentenceChunker` turns that character stream into speakable
sentences at real sentence boundaries.

Both helpers are deliberately conservative: if the model emits plain prose
instead of JSON, the extractor switches to passthrough mode; if a sentence
contains no boundary, the chunker flushes the remainder at the end.
"""

from __future__ import annotations

import re

_JSON_REPLY_START = re.compile(r'"reply"\s*:\s*"')
_BOUNDARY = re.compile(r"[.!?](?=\s|$)")
_ABBREVIATIONS = (
    "rs.",
    "inr.",
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "no.",
    "approx.",
    "e.g.",
    "i.e.",
    "vs.",
    "etc.",
)

_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class ReplyStreamExtractor:
    """Extract the ``reply`` text from streamed JSON tokens."""

    def __init__(self) -> None:
        self.buffer = ""
        self.mode: str | None = None
        self._position = 0
        self._reply_started = False
        self._reply_done = False
        self._escape = False
        self._unicode: str | None = None

    @property
    def reply_started(self) -> bool:
        return self._reply_started or self.mode == "plain"

    @property
    def reply_done(self) -> bool:
        return self._reply_done or self.mode == "plain"

    def feed(self, token: str) -> str:
        """Consume one token and return newly decoded reply characters."""

        if not token:
            return ""

        self.buffer += token

        if self.mode is None:
            stripped = self.buffer.lstrip()

            if not stripped:
                return ""

            if stripped.startswith("{") or '"reply"' in stripped[:120]:
                self.mode = "json"
            elif len(stripped) >= 12 or any(ch in stripped for ch in ".!?"):
                # Plain-prose fallback. Emit everything accumulated so far;
                # later tokens pass straight through.
                self.mode = "plain"
                return self.buffer
            else:
                return ""

        if self.mode == "plain":
            return token

        if self._reply_done:
            return ""

        if not self._reply_started:
            match = _JSON_REPLY_START.search(self.buffer)
            if match is None:
                return ""
            self._reply_started = True
            self._position = match.end()

        return self._decode_reply()

    def _decode_reply(self) -> str:
        output: list[str] = []

        while self._position < len(self.buffer):
            char = self.buffer[self._position]

            if self._unicode is not None:
                self._unicode += char
                self._position += 1
                if len(self._unicode) == 4:
                    try:
                        output.append(chr(int(self._unicode, 16)))
                    except ValueError:
                        pass
                    self._unicode = None
                continue

            if self._escape:
                self._escape = False

                if char == "u":
                    self._unicode = ""
                    self._position += 1
                    continue

                output.append(_ESCAPES.get(char, char))
                self._position += 1
                continue

            if char == "\\":
                self._escape = True
                self._position += 1
                continue

            if char == '"':
                self._reply_done = True
                self._position += 1
                break

            output.append(char)
            self._position += 1

        return "".join(output)

    def flush(self) -> str:
        """Return any plain-prose buffer not emitted yet."""

        if self.mode is None and self.buffer.strip():
            self.mode = "plain"
            return self.buffer
        return ""


class SentenceChunker:
    """Group streamed reply characters into speakable sentences."""

    def __init__(self, min_chars: int = 8, max_chars: int = 240) -> None:
        self.min_chars = max(1, int(min_chars))
        self.max_chars = max(self.min_chars, int(max_chars))
        self.buffer = ""

    @staticmethod
    def _is_abbreviation(candidate: str) -> bool:
        lowered = candidate.casefold().rstrip()
        return any(lowered.endswith(abbreviation) for abbreviation in _ABBREVIATIONS)

    def feed(self, text: str) -> list[str]:
        """Return sentences completed by *text* (may be empty)."""

        if text:
            self.buffer += text

        sentences: list[str] = []

        while True:
            chosen = None

            for match in _BOUNDARY.finditer(self.buffer):
                candidate = self.buffer[: match.end()].strip()

                if not candidate or self._is_abbreviation(candidate):
                    continue

                if len(candidate) < self.min_chars:
                    # Too short to be a useful utterance; keep scanning for
                    # a later boundary so "Yes. The fee is ..." stays whole.
                    continue

                chosen = (candidate, match.end())
                break

            if chosen is None:
                break

            sentence, end = chosen
            sentences.append(sentence)
            self.buffer = self.buffer[end:].lstrip()

        # A model that never emits punctuation must not stall TTS forever.
        while len(self.buffer) > self.max_chars:
            split_at = self.buffer.rfind(" ", 0, self.max_chars)
            if split_at <= 0:
                split_at = self.max_chars
            sentences.append(self.buffer[:split_at].strip())
            self.buffer = self.buffer[split_at:].lstrip()

        return [sentence for sentence in sentences if sentence]

    def flush(self) -> str:
        """Return the unterminated remainder."""

        remainder = self.buffer.strip()
        self.buffer = ""
        return remainder
