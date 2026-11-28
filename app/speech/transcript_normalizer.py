"""Phonetic transcript repair for short kiosk questions.

Whisper tiny (and even base) frequently mishears a short, quiet kiosk
question: ``"Who are you?"`` at low RMS often comes back as ``"Who love
you?"``. Retrieval and the LLM then answer a different question entirely.

The map below is deliberately tiny and phrase-level. Every entry is a
whole-phrase rewrite of a mishearing that has no sensible meaning in an
admission-kiosk context, so a correct transcript is never altered:

    "who love you"       -> "who are you"
    "who loves you"      -> "who are you"
    "who am eye"         -> "who am i"

Callers keep the raw transcript for logs and events, so every repair stays
visible and auditable.
"""

from __future__ import annotations

import re

# (pattern, replacement). Patterns are phrase-anchored with word
# boundaries and must not collide with a legitimate question.
_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwho\s+loves?\s+(?:you|u)\b", re.IGNORECASE), "who are you"),
    (re.compile(r"\bwho\s+(?:our|r)\s+(?:you|u)\b", re.IGNORECASE), "who are you"),
    (re.compile(r"\bwho\s+are\s+(?:u|ya)\b", re.IGNORECASE), "who are you"),
    (
        re.compile(
            r"\bdo\s+you\s+(?:recognise|recognize|know|remember)\s+me\b",
            re.IGNORECASE,
        ),
        "do you recognize me",
    ),
    (
        re.compile(r"\bcan\s+you\s+see\s+(?:me|us)\b", re.IGNORECASE),
        "can you see me",
    ),
    (re.compile(r"\bwho\s+am\s+(?:i|eye)\b", re.IGNORECASE), "who am i"),
)


def normalize_transcript(text: str) -> tuple[str, list[str]]:
    """Return ``(normalized_text, applied_repairs)``.

    ``applied_repairs`` is empty when the transcript was already clean, so
    callers can log a repair only when one actually happened.
    """

    if not text or not text.strip():
        return text, []

    repaired = text
    applied: list[str] = []

    for pattern, replacement in _ALIASES:
        match = pattern.search(repaired)

        if match is None:
            continue

        # Preserve sentence case: "Who love you?" -> "Who are you?".
        if match.group(0)[:1].isupper():
            replacement = replacement[:1].upper() + replacement[1:]

        # The canonical phrase matches its own pattern; that is not a
        # repair and must not be logged as one.
        if match.group(0).casefold() == replacement.casefold():
            continue

        applied.append("%r -> %r" % (match.group(0), replacement))
        repaired = pattern.sub(replacement, repaired)

    # Collapse whitespace introduced by a rewrite while preserving the
    # original casing of untouched words.
    repaired = re.sub(r"\s+", " ", repaired).strip()

    return repaired, applied
