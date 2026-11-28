"""Deterministic query alias expansion for Greeto retrieval.

Campus visitors speak in shorthand and Whisper frequently transcribes branch
names as homophones (``hostile`` for ``hostel``, ``up tech`` for ``UPTAC``).
Expanding those tokens before lexical and dense retrieval dramatically
improves recall for a tiny corpus, while leaving the *raw* transcript that is
sent to the language model untouched.

The expansion is deliberately single-pass: replacements are applied against
the original string with one combined regular expression, so inserted text
(``its engineering college``) is never expanded again.
"""

from __future__ import annotations

import re

# Ordered longest/most-specific first. Every short branch code is word
# bounded so "docs" never becomes "do computer science and engineering".
_ALIAS_TABLE: tuple[tuple[str, str], ...] = (
    (r"artificial intelligence\s*(?:and|&|/)\s*machine learning", "artificial intelligence and machine learning"),
    (r"\bai\s*[/&-]?\s*ml\b", "artificial intelligence and machine learning"),
    (r"\baiml\b", "artificial intelligence and machine learning"),
    (r"\bcomputer\s*science\s*(?:and|&)\s*engineering\b", "computer science and engineering"),
    (r"\belectronics\s*(?:and|&)\s*communication\s*engineering\b", "electronics and communication engineering"),
    (r"\belectrical\s*(?:and|&)\s*computer\s*engineering\b", "electrical and computer engineering"),
    (r"\bmechanical\s*engineering\b", "mechanical engineering"),
    (r"\bcivil\s*engineering\b", "civil engineering"),
    (r"\bdata\s*science\b", "data science"),
    (r"\btransfer\s*certificate\b", "transfer certificate"),
    (r"\bbtech\s*lateral\s*entry(?:\s*second\s*year)?\b", "btech lateral entry second year"),
    (r"\blateral\s*entry\b", "btech lateral entry second year"),
    (r"\bi\s*t\s*s\b", "its engineering college"),
    (r"\bits\b", "its engineering college"),
    (r"\bit'?s\b", "its engineering college"),
    (r"\ba\s*k\s*t\s*u\b", "aktu dr apj abdul kalam technical university"),
    (r"\baktu\b", "aktu dr apj abdul kalam technical university"),
    (r"\bupsee\b", "aktu dr apj abdul kalam technical university"),
    (r"\bup\s*tech\b", "uptac counseling"),
    (r"\buptac\b", "uptac counseling"),
    (r"\bfee\s*structure\b", "fee structure tuition charges discount refund"),
    (r"\bcse\b", "computer science and engineering"),
    (r"\bcs\b", "computer science and engineering"),
    (r"\bece\b", "electronics and communication engineering"),
    (r"\bec\b", "electrical and computer engineering"),
    (r"\bee\b", "electrical and computer engineering"),
    (r"\bme\b", "mechanical engineering"),
    (r"\bce\b", "civil engineering"),
    (r"\bds\b", "data science"),
    (r"\bai\b", "artificial intelligence and machine learning"),
    (r"\btc\b", "transfer certificate"),
    (r"\ble\b", "btech lateral entry second year"),
    (r"\bhostile\b", "hostel"),
    (r"\bfees\b", "fees fee structure tuition charges discount refund"),
)

# "me" is both the Mechanical Engineering shorthand and an English pronoun.
# These common conversational patterns are masked before expansion and
# restored afterwards so "tell me about the campus" is not mangled.
_ME_PROTECTION = re.compile(
    r"(\b(?:tell|show|give|help|let|for|to|with|about|from|of|by|at|in)\s+)me\b",
    re.IGNORECASE,
)

# The placeholder must not contain the letters "me", otherwise the
# \bme\b alias would match inside it during the combined substitution.
_ME_PLACEHOLDER = "\u0000pr0noun\u0000"

# Named groups make the replacement unambiguous even when a shorter alias
# could match inside a longer one.
_COMBINED = re.compile(
    "|".join(
        "(?P<g%d>%s)" % (index, pattern)
        for index, (pattern, _) in enumerate(_ALIAS_TABLE)
    ),
    re.IGNORECASE,
)


def _replace(match: re.Match) -> str:
    for index, (_, replacement) in enumerate(_ALIAS_TABLE):
        if match.group("g%d" % index) is not None:
            return replacement
    return match.group(0)


def normalize_query(text: str) -> str:
    """Expand student/parent shorthand and ASR homophones for retrieval.

    The returned string is only used to query the knowledge base. Callers
    must keep the visitor's original transcript for the LLM prompt.
    """

    if not text:
        return text

    protected = _ME_PROTECTION.sub(
        lambda match: match.group(1) + _ME_PLACEHOLDER,
        text,
    )
    expanded = _COMBINED.sub(_replace, protected)
    expanded = expanded.replace(_ME_PLACEHOLDER, "me")

    # Collapse whitespace introduced by replacements, but keep the original
    # casing of untouched words.
    return re.sub(r"\s+", " ", expanded).strip()


# Backwards-friendly alias.
expand_aliases = normalize_query
