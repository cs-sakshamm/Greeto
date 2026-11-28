"""Structured metadata for the Greeto institutional knowledge corpus.

The ``[KB-XX-YY]`` identifiers already encode the domain of each atomic
record. This module turns that identifier into the metadata the retrieval
index and the voice-enrichment pipeline need:

* ``category``          — coarse retrieval domain (fees, admissions, ...);
* ``effective_session`` — the official filing the corpus was verified against;
* ``volatility``        — ``annual`` facts (fees, intake, placements) must be
  re-verified each session, ``static`` facts (address, labs, rules) rarely
  change;
* ``search_aliases``    — deterministic fallback aliases used when the
  generated ``greeto_voice_kb.jsonl`` has not been built yet.

Nothing here is sent to the language model as-is: metadata is retrieval
plumbing, and spoken text is produced separately by
``scripts/enrich_kb_for_voice.py``.
"""

from __future__ import annotations

import re

EFFECTIVE_SESSION = "2026-27"

# Section group -> (category, default volatility)
_GROUP_METADATA: dict[str, tuple[str, str]] = {
    "01": ("identity", "static"),
    "02": ("directory", "static"),
    "03": ("accreditation", "static"),
    "04": ("admissions", "annual"),
    "05": ("departments", "static"),
    "06": ("admissions", "annual"),
    "07": ("admissions", "static"),
    "08": ("fees", "annual"),
    "09": ("scholarships", "annual"),
    "10": ("placements", "annual"),
    "11": ("labs", "static"),
    "12": ("campus", "static"),
    "13": ("regulations", "static"),
}

# Individual records whose volatility differs from their section default.
# Bank details and refund rules are contractual/static even though they live
# in the fee domain; intake numbers and current-year fee schedules are annual.
_VOLATILITY_OVERRIDES: dict[str, str] = {
    "KB-04-05": "static",   # lateral-entry pathway
    "KB-04-07": "static",   # honours registration rules
    "KB-04-08": "static",   # honours eligibility
    "KB-04-09": "static",   # minor degree streams
    "KB-04-10": "static",   # minor degree penalty rule
    "KB-08-05": "static",   # Canara Bank account + IFSC
    "KB-08-06": "static",   # pre-session refund rule
    "KB-08-07": "static",   # post-session refund rule
    "KB-08-08": "static",   # counseling portal refund rule
    "KB-08-10": "static",   # 14-day APERP refund SOP
    "KB-06-02": "static",   # eligibility percentages
    "KB-06-03": "static",   # compulsory Class 12 subjects
    "KB-06-05": "static",   # MBA eligibility + accepted exams
    "KB-09-04": "static",   # defence scholarship slabs
    "KB-09-05": "static",   # sports scholarship slabs
    "KB-09-07": "static",   # lateral-entry exclusion / loan guidance
}

_KB_ID_PATTERN = re.compile(r"KB-(\d{2})-(\d{2})", re.IGNORECASE)

# Compact, deterministic aliases. These keep retrieval useful even before the
# voice corpus has been regenerated; the generated per-record aliases are
# appended on top of these.
_ALIAS_OVERRIDES: dict[str, list[str]] = {
    "KB-02-01": ["college address", "pin code 201310", "knowledge park 3 location"],
    "KB-02-07": ["director name", "who is the director", "mayank garg"],
    "KB-02-08": ["chairman name", "r p chadha", "who is the chairman"],
    "KB-02-11": ["registrar contact", "registrar phone", "admin office number"],
    "KB-02-13": ["how many faculty", "194 faculty", "26 phds 10 iitians"],
    "KB-03-03": ["nba accredited branches", "which branch is nba", "ece electrical nba"],
    "KB-03-04": ["naac accreditation", "naac cycles", "naac 2024 29"],
    "KB-03-06": ["nirf innovation rank", "ariia band a", "nirf 151 300"],
    "KB-04-03": ["seat intake", "how many seats", "branch wise intake"],
    "KB-04-08": ["honours eligibility", "7.5 ygpa", "honours degree requirement"],
    "KB-04-10": ["minor degree penalty", "18 20 credits minor", "minor not completed"],
    "KB-05-08": ["skill hours", "310 hours", "210 hours cse"],
    "KB-06-02": ["btech eligibility", "45 percent pcm", "40 percent sc st"],
    "KB-06-05": ["mba eligibility", "cat mat xat", "50 percent graduation"],
    "KB-07-01": ["documents required", "admission checklist", "12 documents"],
    "KB-08-01": ["mba total fee", "256500", "mba fee installments"],
    "KB-08-03": ["civil ece mechanical discount", "20000 branch discount", "fee concession in core branches"],
    "KB-08-04": ["aktu fees", "2306 7500", "university charges", "aktu pre enrollment examination fees"],
    "KB-08-05": ["canara bank details", "ifsc cnrb0018556", "account number 85563060000038"],
    "KB-08-06": ["pre session refund", "1000 deduction", "cancel admission refund"],
    "KB-08-10": ["aperp refund", "14 days refund", "security refund process"],
    "KB-09-04": ["defence scholarship", "30000 75000", "ex servicemen scholarship"],
    "KB-09-05": ["sports scholarship", "cultural scholarship", "zonal state national scholarship"],
    "KB-09-07": ["lateral entry scholarship", "education loan", "vidyalakshmi"],
    "KB-10-01": ["placement highlights", "49 lpa 4.5 lpa", "highest average package"],
    "KB-11-01": ["intel unnati lab", "dual a40 48gb", "xeon lab"],
    "KB-11-02": ["apple ios lab", "m3 mac lab", "apple lab"],
    "KB-12-03": ["library volumes", "central library books", "e library"],
    "KB-12-04": ["hostel facility", "boys girls hostel", "hostel rooms"],
    "KB-12-08": ["bus transport", "college bus routes", "transport facility"],
    "KB-13-01": ["attendance rule", "75 percent attendance", "condonation"],
    "KB-13-03": ["sessional exams", "pre university test", "internal exams"],
    "KB-13-04": ["passing marks", "grading rules", "30 40 50 percent"],
    "KB-13-09": ["anti ragging", "ehasas udghosh", "grievance committee"],
}


def parse_kb_id(value: str | None) -> tuple[str | None, str | None]:
    """Return ``(kb_id, group)`` for a record identifier."""

    if not value:
        return None, None

    match = _KB_ID_PATTERN.search(str(value))
    if not match:
        return None, None

    return "KB-%s-%s" % (match.group(1), match.group(2)), match.group(1)


def category_for(kb_id: str | None) -> str:
    _, group = parse_kb_id(kb_id)
    return _GROUP_METADATA.get(group or "", ("general", "static"))[0]


def volatility_for(kb_id: str | None) -> str:
    normalized, group = parse_kb_id(kb_id)
    if normalized and normalized in _VOLATILITY_OVERRIDES:
        return _VOLATILITY_OVERRIDES[normalized]
    return _GROUP_METADATA.get(group or "", ("general", "static"))[1]


def default_aliases(kb_id: str | None) -> list[str]:
    normalized, _ = parse_kb_id(kb_id)
    if not normalized:
        return []
    return list(_ALIAS_OVERRIDES.get(normalized, []))


def metadata_for(kb_id: str | None) -> dict:
    """Return the complete structured metadata block for one record."""

    normalized, _ = parse_kb_id(kb_id)
    if not normalized:
        return {
            "category": "general",
            "effective_session": EFFECTIVE_SESSION,
            "volatility": "static",
            "search_aliases": [],
        }

    return {
        "kb_id": normalized,
        "category": category_for(normalized),
        "effective_session": EFFECTIVE_SESSION,
        "volatility": volatility_for(normalized),
        "search_aliases": default_aliases(normalized),
    }
