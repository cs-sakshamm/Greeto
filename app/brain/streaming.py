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

# Prompt scaffolding a small instruct model sometimes echoes into the
# ``reply`` field (``Your answer: ...``, ``Assistant: ...``). It is not part
# of the answer and must never reach TTS, captions, logs or history.
_SCAFFOLD_LABELS = (
    "your answer",
    "answer",
    "assistant",
    "greeto",
    "response",
    "reply",
    "model",
    "ai",
    "q",
    "a",
)

_SCAFFOLD_PREFIX = re.compile(
    r"^\s*(?:%s)\s*:\s*" % "|".join(_SCAFFOLD_LABELS),
    re.IGNORECASE,
)


# A small instruct model sometimes copies a format example verbatim
# ("short spoken answer", "<reply>"). Those are instructions, never an
# answer, so they are removed before TTS, captions or history see them.
_PLACEHOLDER_TOKENS = re.compile(
    r"<\s*(?:short\s+spoken\s+answer|your\s+answer|spoken\s+answer|"
    r"reply|answer|response)\s*>",
    re.IGNORECASE,
)

_PLACEHOLDER_VALUES = frozenset(
    {
        "short spoken answer",
        "short spoken response",
        "spoken answer",
        "your answer",
        "reply",
        "response",
        "answer",
        "placeholder",
    }
)


def strip_placeholders(text: str) -> str:
    """Remove prompt placeholders echoed as if they were the answer.

    A response that consists only of a placeholder collapses to an empty
    string, which makes the caller use its normal fallback instead of
    speaking the instruction aloud. A genuine sentence that merely
    contains a bracketed word is left intact.
    """

    if not text:
        return text

    cleaned = _PLACEHOLDER_TOKENS.sub("", text)

    plain = re.sub(r"\s+", " ", cleaned.strip().strip("<>\"'` ")).casefold()

    if plain in _PLACEHOLDER_VALUES:
        return ""

    return cleaned.strip()


def strip_reply_prefix(text: str) -> str:
    """Remove leading scaffolding labels such as ``Your answer:``.

    Labels can stack (``Greeto: Answer: ...``), so the match is repeated
    until the text stops changing. Text that does not start with a label is
    returned unchanged.
    """

    if not text:
        return text

    cleaned = text

    while True:
        updated = _SCAFFOLD_PREFIX.sub("", cleaned, count=1)

        if updated == cleaned:
            break

        cleaned = updated

    return cleaned


class ReplyPrefixFilter:
    """Strip a scaffolding label from the first characters of a stream.

    A token-by-token stream cannot decide on the first character whether
    ``A`` starts ``Answer:`` or the word ``A``, so the head of the reply is
    held until a colon, newline, the lookahead limit or a clearly
    non-scaffolding word makes the decision safe. After that the filter is
    a passthrough and adds no latency to the rest of the answer.
    """

    def __init__(self, lookahead: int = 24) -> None:
        self.lookahead = max(4, int(lookahead))
        self.buffer = ""
        self._settled = False

    @staticmethod
    def _may_be_scaffold(head: str) -> bool:
        """True while *head* could still grow into a known label."""

        trimmed = head.rstrip()

        if not trimmed:
            return True

        for label in _SCAFFOLD_LABELS:
            if (label + ":").startswith(trimmed):
                return True

        return False

    @staticmethod
    def _may_be_placeholder(head: str) -> bool:
        """True while *head* could still grow into a placeholder value.

        Token boundaries are arbitrary, so ``<short spoken answer>`` can
        arrive as ``<short`` + `` spoken answer>``. The head is held until
        the closing bracket or until it can no longer be a placeholder, so
        the whole value is removed instead of leaking a fragment.
        """

        lowered = head.lstrip().casefold()

        if not lowered:
            return True

        if lowered.startswith("<"):
            return ">" not in lowered

        for value in _PLACEHOLDER_VALUES:
            if value.startswith(lowered.strip()):
                return True

        return False

    def feed(self, text: str) -> str:
        """Return the text that is safe to emit now (may be empty)."""

        if not text:
            return ""

        if self._settled:
            return text

        self.buffer += text

        head = self.buffer.lstrip()
        window = head[: self.lookahead]

        decided = (
            len(head) >= self.lookahead
            or ":" in window
            or "\n" in window
            or (
                not self._may_be_scaffold(window.casefold())
                and not self._may_be_placeholder(window)
            )
        )

        if not decided:
            return ""

        self._settled = True
        emitted = strip_placeholders(strip_reply_prefix(self.buffer))
        self.buffer = ""

        return emitted

    def flush(self) -> str:
        """Emit anything still held back at the end of the stream."""

        if self._settled:
            self.buffer = ""
            return ""

        self._settled = True
        emitted = strip_placeholders(strip_reply_prefix(self.buffer))
        self.buffer = ""

        return emitted


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
        self._prefix_filter = ReplyPrefixFilter()

    @property
    def reply_started(self) -> bool:
        return self._reply_started or self.mode == "plain"

    @property
    def reply_done(self) -> bool:
        return self._reply_done or self.mode == "plain"

    def feed(self, token: str) -> str:
        """Consume one token and return newly decoded reply characters."""

        return self._prefix_filter.feed(self._feed_raw(token))

    def _feed_raw(self, token: str) -> str:
        """Decode one token; scaffolding prefix filtering happens in feed()."""

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

        raw = ""

        if self.mode is None and self.buffer.strip():
            self.mode = "plain"
            raw = self.buffer

        return self._prefix_filter.feed(raw) + self._prefix_filter.flush()


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
