#!/usr/bin/env python3
"""Generate a voice-ready Greeto corpus from the 113 atomic KB records.

The source Markdown is never modified. For every ``[KB-XX-XX]`` record this
script emits one JSONL row containing:

    kb_id, category, effective_session, volatility
    question, answer              searchable / quotable text (numbers intact)
    spoken_answer                 TTS-normalised, bullet-free, 35-55 words
    search_aliases                3-5 noisy ASR / colloquial queries
    spoken_word_count, needs_review

The output is ``knowledge_base/greeto_voice_kb.jsonl``. ``vector_store.py``
detects that file automatically, indexes ``search_aliases`` for lexical and
dense retrieval, and keeps ``spoken_answer`` out of the model prompt.

Usage:

    .venv/Scripts/python.exe scripts/enrich_kb_for_voice.py
    .venv/Scripts/python.exe scripts/enrich_kb_for_voice.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.brain.kb_metadata import metadata_for  # noqa: E402
from app.brain.vector_store import load_chunks  # noqa: E402

DEFAULT_SOURCE = "knowledge_base/knowledge_base.md"
DEFAULT_OUTPUT = "knowledge_base/greeto_voice_kb.jsonl"

MAX_SPOKEN_WORDS = 60
TARGET_SPOKEN_WORDS = 55

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "what", "which", "who", "whom", "whose", "how", "why", "when", "where",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "i", "me", "my", "we", "our", "you", "your", "of", "to", "for", "in",
    "on", "at", "by", "with", "and", "or", "about", "please", "tell", "give",
    "there", "it", "this", "that", "these", "those", "have", "has", "had",
    "from", "as", "if", "any", "much", "many", "know", "want", "need",
    "like", "college", "its", "i.t.s", "at", "into", "across", "during",
    "also", "per", "their", "they", "them", "not", "no", "yes", "all",
}

_ASR_REPLACEMENTS = (
    ("engineering", "enginering"),
    ("computer", "compuer"),
    ("mechanical", "mechnical"),
    ("electrical", "electrical"),
    ("electronics", "electronics"),
    ("communication", "comunication"),
    ("admission", "addmission"),
    ("scholarship", "scholership"),
    ("placement", "placemnt"),
    ("attendance", "attendence"),
    ("hostel", "hostal"),
    ("fees", "fess"),
    ("fee", "fess"),
    ("eligibility", "eligibilty"),
    ("percentage", "percent"),
    ("director", "dierctor"),
    ("registrar", "registerar"),
    ("certificate", "certificat"),
    ("accreditation", "accrediation"),
    ("library", "libary"),
    ("transport", "transprot"),
    ("document", "documnet"),
)

_HINGLISH_BY_CATEGORY = {
    "fees": "fees kitni hai",
    "admissions": "admission kaise le",
    "placements": "placement kitna hai",
    "scholarships": "scholarship milegi kya",
    "directory": "contact number kya hai",
    "campus": "college kaisa hai",
    "labs": "lab facility kaisi hai",
    "regulations": "attendance rule kya hai",
    "accreditation": "college accredited hai kya",
    "departments": "department kaisa hai",
    "identity": "tum kaun ho",
}


def _indian_amount_words(value: int) -> str:
    """Format an Indian rupee amount the way a person speaks it."""

    if value >= 10_000_000:
        crore, remainder = divmod(value, 10_000_000)
        parts = ["%d crore" % crore]
        lakh = remainder // 100_000
        if lakh:
            parts.append("%d lakh" % lakh)
        return " ".join(parts)

    if value >= 100_000:
        lakh, remainder = divmod(value, 100_000)
        thousand, hundreds = divmod(remainder, 1000)
        parts = ["%d lakh" % lakh]
        if thousand:
            parts.append("%d thousand" % thousand)
        if hundreds:
            parts.append(str(hundreds))
        return " ".join(parts)

    return "{:,}".format(value)


def _currency_amount(match: re.Match) -> str:
    raw = match.group(1).replace(",", "")
    if not raw.isdigit():
        return match.group(0)
    return _indian_amount_words(int(raw)) + " rupees"


def _bare_amount(match: re.Match) -> str:
    raw = match.group(1).replace(",", "")
    if not raw.isdigit():
        return match.group(0)
    value = int(raw)
    if value >= 100_000:
        return _indian_amount_words(value)
    return match.group(0)


def spoken_normalize(text: str) -> str:
    """Convert an answer into clean, speakable text for Piper TTS.

    Searchable text keeps ``₹2,56,500``, ``49 LPA``, ``CNRB0018556`` and
    ``1800-180-0840`` exactly as written. Only this spoken copy is
    number/symbol normalised.
    """

    if not text:
        return ""

    spoken = text

    # Markdown noise and parenthetical clutter.
    spoken = re.sub(r"^[ \t]*[-*•#]+\s*", "", spoken, flags=re.MULTILINE)
    spoken = spoken.replace("*", "").replace("`", "").replace("_", "")
    spoken = spoken.replace("(", "").replace(")", "")
    spoken = spoken.replace("[", "").replace("]", "")

    # Abbreviations that Piper otherwise spells oddly.
    spoken = re.sub(r"\bB\.\s*Tech\b", "B Tech", spoken, flags=re.IGNORECASE)
    spoken = re.sub(r"\bM\.\s*B\.\s*A\.?\b", "MBA", spoken, flags=re.IGNORECASE)
    spoken = re.sub(r"\bPh\.\s*D\.?", "PhD", spoken, flags=re.IGNORECASE)
    spoken = re.sub(r"\be\.g\.", "for example", spoken, flags=re.IGNORECASE)
    spoken = re.sub(r"\bi\.e\.", "that is", spoken, flags=re.IGNORECASE)

    # Academic years. Do this before phone-number formatting so an en dash
    # or a hyphen between years is not mistaken for a phone separator.
    spoken = re.sub(
        r"\b((?:19|20)\d{2})\s*[-\u2013]\s*(\d{2})\b",
        r"\1 to \2",
        spoken,
    )

    # Money first, while the currency marker still identifies the amount.
    spoken = re.sub(r"(?:\u20b9|Rs\.?|INR)\s*([\d,]+)", _currency_amount, spoken, flags=re.IGNORECASE)
    spoken = re.sub(r"\b(\d{1,3}(?:,\d{2,3})+)\b", _bare_amount, spoken)

    # Package values and percentages.
    spoken = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*LPA\b",
        lambda match: "%s lakhs per annum" % match.group(1),
        spoken,
        flags=re.IGNORECASE,
    )
    spoken = re.sub(r"\s*%", " percent", spoken)
    spoken = re.sub(r"\b(\d+)\s*:\s*(\d+)\b", r"\1 to \2", spoken)

    # Phone numbers: keep the digits, drop the dashes so Piper pauses
    # naturally instead of reading "one eight hundred dash".
    spoken = re.sub(
        r"\b(\d{3,5})[- ](\d{3})[- ](\d{4})\b",
        r"\1 \2 \3",
        spoken,
    )

    # Remaining slashes and ampersands.
    spoken = re.sub(r"\s*/\s*", " or ", spoken)
    spoken = spoken.replace("&", " and ")
    spoken = spoken.replace("\u2013", " to ")

    # Hygiene: collapse whitespace, remove space before punctuation, and
    # make sure every sentence ends cleanly.
    spoken = re.sub(r"\s+", " ", spoken).strip()
    spoken = re.sub(r"\s+([.,;:!?])", r"\1", spoken)
    spoken = re.sub(r",\s*,", ",", spoken).strip(" ,;:")

    if spoken and not spoken.endswith((".", "!", "?")):
        spoken += "."

    return spoken


def _truncate_spoken(text: str) -> tuple[str, bool]:
    """Keep 2-3 sentences / ~35-55 words and report whether it was cut."""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return "", False

    original_words = len(text.split())
    kept: list[str] = []
    word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if kept and (word_count + sentence_words > TARGET_SPOKEN_WORDS or len(kept) >= 3):
            break
        kept.append(sentence)
        word_count += sentence_words

    spoken = " ".join(kept).strip()

    if len(spoken.split()) > MAX_SPOKEN_WORDS:
        spoken = " ".join(spoken.split()[:TARGET_SPOKEN_WORDS]).rstrip(",;:") + "."

    if spoken and not spoken.endswith((".", "!", "?")):
        spoken += "."

    return spoken, original_words > MAX_SPOKEN_WORDS


def _keyword_phrase(question: str, limit: int = 6) -> str:
    tokens = re.findall(r"[a-z0-9][a-z0-9,./-]*", question.casefold())
    kept = []
    for token in tokens:
        cleaned = token.strip(".,/-")
        if len(cleaned) < 2 or cleaned in _STOPWORDS:
            continue
        kept.append(cleaned)
    return " ".join(kept[:limit])


def _noisy_alias(question: str) -> str:
    text = question.casefold()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    for source, target in _ASR_REPLACEMENTS:
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _number_alias(question: str, answer: str) -> str:
    """Pair a distinctive number with a topic word from the question."""

    values = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", answer)
    distinctive = [
        value
        for value in values
        if len(value.replace(",", "")) >= 4 or "," in value
    ]
    if not distinctive:
        return ""
    topic = _keyword_phrase(question, limit=3)
    if not topic:
        return distinctive[0]
    return "%s %s" % (topic, distinctive[0].strip(","))


def build_aliases(question: str, answer: str, kb_id: str, category: str) -> list[str]:
    metadata = metadata_for(kb_id)
    candidates: list[str] = list(metadata["search_aliases"])

    keyword = _keyword_phrase(question)
    if keyword:
        candidates.append(keyword)

    noisy = _noisy_alias(question)
    if noisy and noisy != keyword:
        candidates.append(noisy)

    hinglish = _HINGLISH_BY_CATEGORY.get(category)
    if hinglish:
        candidates.append(hinglish)

    number = _number_alias(question, answer)
    if number:
        candidates.append(number)

    aliases: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate).strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        aliases.append(normalized)

    return aliases[:5]


def _parse_qa(text: str) -> tuple[str, str] | None:
    match = re.match(r"Q:\s*(.*?)\s*\nA:\s*(.*)\Z", text, re.DOTALL)
    if not match:
        return None
    question = re.sub(r"\s+", " ", match.group(1)).strip()
    answer = re.sub(r"\s+", " ", match.group(2)).strip()
    if not question or not answer:
        return None
    return question, answer


def build_records(source: Path) -> list[dict]:
    chunks = load_chunks(source)
    records: list[dict] = []
    seen: set[str] = set()

    for chunk in chunks:
        if chunk.kind != "qa" or not chunk.qa_id:
            continue

        kb_id = chunk.kb_id or chunk.qa_id
        if kb_id in seen:
            continue

        parsed = _parse_qa(chunk.text)
        if parsed is None:
            continue
        question, answer = parsed

        metadata = metadata_for(kb_id)
        spoken = spoken_normalize(answer)
        spoken, needs_review = _truncate_spoken(spoken)
        aliases = build_aliases(question, answer, kb_id, metadata["category"])

        records.append(
            {
                "kb_id": kb_id,
                "qa_id": kb_id,
                "category": metadata["category"],
                "effective_session": metadata["effective_session"],
                "volatility": metadata["volatility"],
                "question": question,
                "answer": answer,
                "spoken_answer": spoken,
                "spoken_word_count": len(spoken.split()),
                "needs_review": needs_review,
                "search_aliases": aliases,
                "source": source.name,
            }
        )
        seen.add(kb_id)

    records.sort(key=lambda row: row["kb_id"])
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the generated corpus without rewriting it",
    )
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.is_absolute():
        source = PROJECT_ROOT / source

    if not source.exists():
        print("Knowledge source not found: %s" % source)
        return 1

    records = build_records(source)
    if not records:
        print("No [KB-XX-XX] records found in %s" % source)
        return 1

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    if not args.check:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(records)
    long_flags = sum(1 for record in records if record["needs_review"])
    short_aliases = sum(1 for record in records if len(record["search_aliases"]) < 3)
    word_counts = [record["spoken_word_count"] for record in records]
    over_limit = sum(1 for count in word_counts if count > MAX_SPOKEN_WORDS)

    print("Records:         %d" % total)
    print("Output:          %s" % output)
    print(
        "Spoken words:    min=%d avg=%.1f max=%d"
        % (
            min(word_counts),
            sum(word_counts) / max(1, len(word_counts)),
            max(word_counts),
        )
    )
    print("Over %d words:   %d" % (MAX_SPOKEN_WORDS, over_limit))
    print("Rewrite flags:   %d" % long_flags)
    print("Aliases < 3:     %d" % short_aliases)

    if total != 113:
        print("WARNING: expected 113 atomic records, found %d" % total)

    if args.check and (over_limit or short_aliases):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
