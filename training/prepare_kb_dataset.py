#!/usr/bin/env python3
"""Build a grounded ChatML dataset from ``knowledge_base/``.

Every atomic ``[KB-XX-XX]`` record becomes one or more training examples:

    system    Greeto's runtime rules (identical to PromptManager)
    user      retrieved college knowledge + a natural/spoken/ASR-noisy question
    assistant the concise spoken answer as the runtime JSON object

The script also emits refusal examples for live or unpublished details
(current-year B.Tech tuition, vacant seats, live cutoffs, out-of-KB
programmes) so a fine-tuned Qwen learns to cite the admission helpline
instead of inventing a number.

Usage:

    .venv/Scripts/python.exe training/prepare_kb_dataset.py
    .venv/Scripts/python.exe training/prepare_kb_dataset.py --answer-format plain

Outputs (gitignored):

    data/qwen25_train.jsonl
    data/qwen25_val.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.brain.prompt_manager import PromptManager  # noqa: E402
from app.brain.vector_store import KBChunk, load_chunks, load_voice_alias_map  # noqa: E402

HELPLINE_ANSWER = (
    "I couldn't find that information in the college database. "
    "Please contact the admission office at 1800-180-0840."
)

# Deliberately unanswerable from the static corpus, or questions built on an
# unverified third-party number. Each entry is (question, context that
# proves the value is not published / not official). Never include the
# aggregator figure itself in the context.
REFUSAL_EXAMPLES = [
    (
        "What is the exact B.Tech tuition fee for the current session?",
        "The official 2026 B.Tech brochure and fee webpage do not publish "
        "itemized monetary B.Tech tuition charges; tuition is regulated by "
        "the Uttar Pradesh State Fee Fixation Committee. The Admission Cell "
        "confirms the active circular on 1800-180-0840.",
    ),
    (
        "How many B.Tech CSE seats are still vacant right now?",
        "Sanctioned intake for B.Tech CSE is 750 seats under the 2026-27 "
        "Mandatory Disclosure. Live vacancy changes every counseling round "
        "and is confirmed by the Admission Cell on 1800-180-0840.",
    ),
    (
        "What was the closing JEE Main cutoff in the latest UPTAC round?",
        "Official cutoffs change every round, and the college does not "
        "publish a static rank table. Contact the Admission Cell on "
        "1800-180-0840 for the current round.",
    ),
    (
        "What is the hostel fee per year?",
        "The official 2026 brochure does not publish itemized hostel charges. "
        "Hostel payment and refund rules are published separately, and the "
        "current circular is confirmed by the Accounts Office.",
    ),
    (
        "What is the salary of the Director?",
        "The directory lists the Director's office phone number and email "
        "address, not personal salary information.",
    ),
    (
        "Do you offer an MBBS programme?",
        "I.T.S Engineering College offers B.Tech programmes and an MBA. "
        "No medical programme is listed in the official programme table.",
    ),
    (
        "Can you guarantee me admission with a 40 percentile JEE Main rank?",
        "Admission is through UPTAC/AKTU counseling or institute-level "
        "direct admission when AKTU eligibility is met and seats are "
        "available. The Admission Cell confirms the current position.",
    ),
    (
        "What is today's live seat availability for MBA?",
        "The MBA sanctioned intake is 60 seats. Live vacancy is not "
        "published in the static knowledge base and is confirmed by the "
        "Admission Cell.",
    ),
    (
        "Is the B.Tech CSE fee 6.10 lakhs?",
        "The college does not publish a total B.Tech package in the static "
        "corpus. Third-party aggregator estimates are not official records; "
        "the Admission Cell confirms the current fee circular on "
        "1800-180-0840.",
    ),
    (
        "Is the B.Tech total fee 4.90 lakhs?",
        "Aggregator totals are not official college records. BTech tuition "
        "is fixed by the UP State Fee Fixation Committee and confirmed by "
        "the Admission Cell on 1800-180-0840.",
    ),
    (
        "Did a student get a 54.74 LPA package?",
        "The official 2024-25 placement record reports a highest B.Tech "
        "package of 49 LPA and an average of 4.5 LPA. Aggregator outliers "
        "are not official records.",
    ),
    (
        "Is the average placement package 18 LPA?",
        "The official placement page reports an overall average package of "
        "4.5 LPA, with 49 LPA highest in B.Tech and 10 LPA highest in MBA. "
        "Contact the CRC for branch-wise medians.",
    ),
    (
        "Is the B.Tech fee 5.29 lakhs?",
        "That figure is a third-party estimate, not an official college "
        "record. Please confirm the active fee circular with the Admission "
        "Cell on 1800-180-0840.",
    ),
    (
        "Is the NAAC grade B++?",
        "The college is NAAC accredited for the 2015-2020 and 2024-2029 "
        "cycles. The official certificate is the only confirmed grade "
        "source; contact the college for verification.",
    ),
    (
        "What is the NAAC CGPA of the college?",
        "The official NAAC certificates are indexed on the college website "
        "for the 2015-2020 and 2024-2029 cycles. The CGPA is verified from "
        "that certificate by the college office.",
    ),
    (
        "What are the Greater Noida bus route charges?",
        "Active bus routes, pickup points and annual transport charges are "
        "confirmed by the Administration Office at admission. They are not "
        "published in the static corpus.",
    ),
    (
        "What is the exact hostel security deposit amount?",
        "Hostel charges, including the security deposit, are confirmed by "
        "the Accounts Office and are not published as a static figure.",
    ),
    (
        "How many MBA seats are vacant right now?",
        "The MBA sanctioned intake is 60 seats. Live vacancy changes every "
        "counseling round and is confirmed by the Admission Cell.",
    ),
    (
        "What is the annual B.Tech tuition for the 2027 batch?",
        "Annual B.Tech tuition is fixed by the UP State Fee Fixation "
        "Committee. The Admission Cell confirms the applicable circular on "
        "1800-180-0840.",
    ),
    (
        "Did a student receive a 1 crore package?",
        "The official 2024-25 record reports a highest B.Tech package of 49 "
        "LPA. Unverified viral figures are not college records.",
    ),
    (
        "What is the hostel fee for the current year?",
        "Itemized hostel charges are not published in the static brochure. "
        "The Administration Office confirms the current hostel circular.",
    ),
    (
        "What is the exact MBA fee for the next session?",
        "The verified official MBA total fee for the 2026-27 brochure is "
        "2,56,500 rupees in four installments. Any next-session revision "
        "is confirmed by the Admission Cell.",
    ),
    (
        "Who is the admission counselor on duty right now?",
        "Live counselor duty rosters are not published in the static "
        "corpus. Please call the admission helpline at 1800-180-0840 or "
        "visit the Admission Cell.",
    ),
    (
        "Is the B.Tech Civil Engineering NBA accredited?",
        "NBA accreditation is active only for B.Tech Electronics and "
        "Communication Engineering and B.Tech Electrical and Computer "
        "Engineering. Civil Engineering is listed as not NBA accredited.",
    ),
    (
        "What is the exact number of companies that visited last week?",
        "The official 2024-25 snapshot reports more than 80 active "
        "recruiters and over 310 offers. A live weekly count is confirmed "
        "by the Corporate Resource Center.",
    ),
    (
        "Is the MBA fee one lakh twenty thousand?",
        "The official MBA Information Brochure 2026 states a total fee of "
        "2,56,500 rupees in four installments. Other figures are not "
        "official college records.",
    ),
]

# Under-specified visitor questions. The answer names the verified figure,
# explains the programme-specific limit, and asks a clarifying question
# instead of inventing one number for both programmes.
AMBIGUOUS_EXAMPLES = [
    {
        "question": "What is the total fee?",
        "answer": (
            "The verified MBA total fee is 2,56,500 rupees in four installments. "
            "Annual B.Tech tuition is fixed by the UP State Fee Fixation "
            "Committee and confirmed by the Admission Cell on 1800-180-0840. "
            "Which programme are you interested in?"
        ),
        "context": (
            "Q: What is the total fee for the MBA program?\n"
            "A: According to the official ITSEC MBA Information Brochure 2026, "
            "the total fee for the two-year MBA program is 2,56,500 rupees, "
            "payable in four installments.\n\n"
            "Q: What is the exact annual B.Tech tuition and hostel fee?\n"
            "A: The official 2026 B.Tech brochure and college fee webpage do not "
            "publish itemized monetary B.Tech tuition or hostel charges, as "
            "tuition is regulated by the Uttar Pradesh State Fee Fixation "
            "Committee. Please contact the Admission Cell on 1800-180-0840."
        ),
    },
    {
        "question": "What is the eligibility?",
        "answer": (
            "For B.Tech, General and OBC candidates need 45 percent in PCM and "
            "SC and ST candidates need 40 percent. For MBA, the requirement is "
            "50 percent for General and OBC or 45 percent for SC and ST. "
            "Which programme are you asking about?"
        ),
        "context": (
            "Q: What is the minimum Class 12 percentage needed for B.Tech admission?\n"
            "A: General and OBC candidates need at least 45 percent aggregate marks "
            "in Physics, Chemistry, and Mathematics in Class 12, while SC and ST "
            "candidates require at least 40 percent in PCM.\n\n"
            "Q: What are the eligibility criteria and accepted exams for MBA admission?\n"
            "A: MBA applicants must hold a bachelor's degree of at least three "
            "years with 50 percent for General and OBC or 45 percent for SC and ST."
        ),
    },
    {
        "question": "How many seats are there?",
        "answer": (
            "Sanctioned intake is 750 seats in B.Tech CSE, 180 in CSE AI and ML, "
            "120 in Data Science, 60 in ECE, 30 each in Electrical and Computer, "
            "Civil and Mechanical, and 60 in MBA. Which branch are you interested in?"
        ),
        "context": (
            "Q: What is the sanctioned seat intake for each B.Tech branch and for MBA?\n"
            "A: Under the 2026-27 Mandatory Disclosure, sanctioned intake is 750 "
            "seats in B.Tech CSE, 180 in CSE AI and ML, 120 in CSE Data Science, "
            "60 in ECE, 30 in Electrical and Computer Engineering, 30 in Civil "
            "Engineering, 30 in Mechanical Engineering, and 60 seats in MBA."
        ),
    },
    {
        "question": "Tell me about placements.",
        "answer": (
            "The official 2024-25 record shows a highest B.Tech package of 49 LPA, "
            "a highest MBA package of 10 LPA, and an overall average of 4.5 LPA. "
            "Would you like B.Tech or MBA placement details?"
        ),
        "context": (
            "Q: What are the official placement highlights at I.T.S Engineering College?\n"
            "A: For the 2024-25 placement drive, the college reported a highest "
            "B.Tech package of 49 LPA, a highest MBA package of 10 LPA, and an "
            "overall average package of 4.5 LPA."
        ),
    },
    {
        "question": "What documents do I need?",
        "answer": (
            "For B.Tech admission, bring Class 10 and 12 mark sheets and "
            "certificates, Migration and Transfer Certificates, a Character "
            "Certificate, Aadhaar, an MBBS medical fitness certificate, four "
            "photographs, and the AKTU allotment letter if applicable. Are you "
            "applying for B.Tech or MBA?"
        ),
        "context": (
            "Q: What documents are required at the time of B.Tech admission?\n"
            "A: You must bring your Class 10 and Class 12 mark sheets and "
            "certificates, original Migration and Transfer Certificates, original "
            "Character Certificate, Aadhaar card, an MBBS-certified Medical "
            "Fitness Certificate, your AKTU allotment letter and counseling fee "
            "receipts if applicable, four recent colored passport photographs, "
            "and any applicable Caste, Income, Domicile, or Gap certificates."
        ),
    },
]

# Natural campus Hinglish, answered in clear English from the same verified
# records. The model must not switch its spoken answer language.
HINGLISH_EXAMPLES = [
    {
        "question": "CSE mein kitni seats hain?",
        "answer": (
            "B.Tech CSE has a sanctioned intake of 750 seats, with 543 actual "
            "admissions recorded in 2024-25. The Admission Cell can confirm the "
            "current counseling position."
        ),
        "qa_id": "KB-04-04",
    },
    {
        "question": "Direct admission ho jayega kya?",
        "answer": (
            "Yes, direct admission is possible if you meet AKTU eligibility: 45 "
            "percent in PCM for General and OBC, or 40 percent for SC and ST. "
            "Please contact the Admission Cell at 1800-180-0840 to check seat "
            "availability."
        ),
        "qa_id": "KB-06-02",
        "extra_qa_ids": ["KB-06-04"],
    },
    {
        "question": "Highest package kitna gaya hai?",
        "answer": (
            "The highest B.Tech package is 49 LPA, the highest MBA package is 10 "
            "LPA, and the overall average package is 4.5 LPA in the official "
            "2024-25 placement record."
        ),
        "qa_id": "KB-10-01",
    },
    {
        "question": "Fees kitni hai?",
        "answer": (
            "The verified MBA total fee is 2,56,500 rupees in four installments. "
            "Annual B.Tech tuition is fixed by the UP State Fee Fixation "
            "Committee and confirmed by the Admission Cell on 1800-180-0840. "
            "Which programme are you interested in?"
        ),
        "qa_id": "KB-08-01",
        "extra_qa_ids": ["KB-08-02"],
    },
    {
        "question": "Hostel ki fees kitni hai?",
        "answer": (
            "Itemized hostel charges are not published in the static brochure. "
            "The Administration Office confirms the current hostel circular, and "
            "the admission helpline is 1800-180-0840."
        ),
        "qa_id": "KB-08-09",
    },
    {
        "question": "Scholarship milegi kya?",
        "answer": (
            "Yes, the college facilitates more than 500 scholarships across "
            "institutional ITS merit and achievement awards, government "
            "scholarships, and corporate scholarships. Eligibility depends on "
            "your programme and category."
        ),
        "qa_id": "KB-09-01",
    },
    {
        "question": "Documents kya chahiye?",
        "answer": (
            "Bring Class 10 and Class 12 mark sheets, Migration and Transfer "
            "Certificates, a Character Certificate, Aadhaar, an MBBS medical "
            "fitness certificate, four photographs, and the AKTU allotment "
            "letter if applicable."
        ),
        "qa_id": "KB-07-01",
    },
    {
        "question": "MBA ka total fee kitna hai?",
        "answer": (
            "The official MBA total fee is 2,56,500 rupees, payable in four "
            "installments."
        ),
        "qa_id": "KB-08-01",
    },
    {
        "question": "Kya main admission le sakta hoon?",
        "answer": (
            "B.Tech admission requires 45 percent in PCM for General and OBC "
            "candidates, or 40 percent for SC and ST. Admission is through UPTAC "
            "counseling or institute-level direct admission when seats are "
            "available."
        ),
        "qa_id": "KB-06-02",
        "extra_qa_ids": ["KB-06-01"],
    },
    {
        "question": "ECE ki cutoff kya hai?",
        "answer": (
            "Official cutoffs change every round, and the college does not "
            "publish a static rank table. Please contact the Admission Cell at "
            "1800-180-0840 for the current cutoff."
        ),
        "qa_id": "KB-06-07",
    },
]

SPOKEN_PREFIXES = (
    "Hi, ",
    "Excuse me, ",
    "Could you please tell me ",
    "I wanted to ask, ",
    "Hello Greeto, ",
)

ASR_REPLACEMENTS = (
    ("what is", "whats"),
    ("what are", "what are"),
    ("i.t.s", "its"),
    ("b.tech", "btech"),
    ("mba", "m b a"),
    ("aktu", "aktoo"),
    ("uptac", "up tac"),
    ("please", "pleas"),
    ("scholarship", "scholarship"),
    ("eligibility", "eligibility"),
    ("hostel", "hostal"),
    ("admission", "addmission"),
    ("percentage", "percent"),
    ("rupees", "rupees"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="knowledge_base")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--max-context-chars", type=int, default=1100)
    parser.add_argument(
        "--answer-format",
        choices=("json", "plain"),
        default="json",
        help="json matches the runtime prompt; plain emits bare answers.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def spoken_variant(question: str, index: int) -> str:
    prefix = SPOKEN_PREFIXES[index % len(SPOKEN_PREFIXES)]
    body = question.strip()
    if prefix.endswith("me ") or prefix.endswith("ask, "):
        body = body[0].lower() + body[1:] if body else body
    return prefix + body


def noisy_variant(question: str, index: int) -> str:
    text = question.casefold()
    text = re.sub(r"[^\w\s,./-]", "", text)
    for source, target in ASR_REPLACEMENTS:
        if index % 3 == 0:
            text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()
    if index % 4 == 0:
        text = text + " please"
    return text


def build_context(target: KBChunk, by_section: dict[str, list[KBChunk]], max_chars: int) -> str:
    blocks = [target.text]

    for chunk in by_section.get(target.section, []):
        if chunk.chunk_id == target.chunk_id or chunk.kind != "qa":
            continue
        candidate = chunk.text
        if len("\n\n".join(blocks + [candidate])) > max_chars:
            continue
        blocks.append(candidate)
        if len(blocks) >= 3:
            break

    return "\n\n".join(blocks)[:max_chars].rstrip()


def assistant_content(answer: str, emotion: str, answer_format: str) -> str:
    if answer_format == "plain":
        return answer
    return json.dumps(
        {"reply": answer, "emotion": emotion, "action": "answer"},
        ensure_ascii=False,
    )


def example(
    system: str,
    context: str,
    question: str,
    answer: str,
    answer_format: str,
    emotion: str = "neutral",
    metadata: dict | None = None,
) -> dict:
    knowledge = context.strip() or "No relevant college knowledge found."
    user = "College knowledge:\n%s\n\nQuestion: %s" % (knowledge, question)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_content(answer, emotion, answer_format)},
        ],
        "metadata": metadata or {},
    }


def pick_emotion(answer: str) -> str:
    lowered = answer.casefold()
    if any(word in lowered for word in ("sorry", "unfortunately", "cannot", "couldn't")):
        return "empathetic"
    if any(word in lowered for word in ("yes", "welcome", "congratulations", "opportunity")):
        return "happy"
    return "neutral"


def build_examples(
    chunks: list[KBChunk],
    max_context_chars: int,
    answer_format: str,
    voice_map: dict[str, dict] | None = None,
):
    qa_chunks = [chunk for chunk in chunks if chunk.kind == "qa" and chunk.qa_id]
    by_section: dict[str, list[KBChunk]] = {}
    by_qa: dict[str, KBChunk] = {}
    for chunk in chunks:
        by_section.setdefault(chunk.section, []).append(chunk)
        if chunk.qa_id:
            by_qa[chunk.qa_id] = chunk

    voice_map = voice_map or {}

    system = PromptManager.system_prompt(disable_thinking=False)
    examples: list[dict] = []

    for chunk in qa_chunks:
        match = re.search(r"Q:\s*(.*?)\nA:\s*(.*)", chunk.text, re.DOTALL)
        if not match:
            continue
        question, answer = match.group(1).strip(), match.group(2).strip()
        answer = re.sub(r"\s+", " ", answer).strip()
        if not question or not answer:
            continue

        voice = voice_map.get(chunk.qa_id or "", {})
        spoken_answer = (voice.get("spoken_answer") or answer).strip()
        aliases = voice.get("search_aliases") or chunk.search_aliases or []

        context = build_context(chunk, by_section, max_context_chars)
        metadata = {"qa_id": chunk.qa_id, "section": chunk.section, "kind": "grounded"}

        variants = [
            question,
            spoken_variant(question, len(examples)),
            noisy_variant(question, len(examples)),
        ]

        if aliases:
            variants.append("Please tell me about %s." % aliases[0].rstrip("."))

        variants.append("Quick question: %s" % question.rstrip("?").lower() + "?")

        for variant in dict.fromkeys(variants):
            examples.append(
                example(
                    system,
                    context,
                    variant,
                    spoken_answer,
                    answer_format,
                    emotion=pick_emotion(spoken_answer),
                    metadata=dict(metadata, variant=variant),
                )
            )

    def context_for(qa_ids: list[str], fallback: str = "") -> str:
        blocks: list[str] = []
        for qa_id in qa_ids:
            chunk = by_qa.get(qa_id)
            if chunk is None:
                continue
            candidate = chunk.text
            if blocks and len("\n\n".join(blocks + [candidate])) > max_context_chars:
                continue
            blocks.append(candidate)

        combined = "\n\n".join(blocks)[:max_context_chars].rstrip()
        return combined or fallback[:max_context_chars]

    # Under-specified questions: state the verified facts, name the limit,
    # and ask which programme the visitor means.
    for item in AMBIGUOUS_EXAMPLES:
        question = item["question"]
        answer = item["answer"]
        context = item.get("context") or ""
        base = len(examples)
        variants = [
            question,
            spoken_variant(question, base),
            noisy_variant(question, base),
        ]
        for variant in dict.fromkeys(variants):
            examples.append(
                example(
                    system,
                    context,
                    variant,
                    answer,
                    answer_format,
                    emotion="neutral",
                    metadata={"kind": "ambiguous", "variant": variant},
                )
            )

    # Hinglish question, English answer. Grounded in the same verified QA
    # records, including the follow-up context when two records are needed.
    for item in HINGLISH_EXAMPLES:
        question = item["question"]
        answer = item["answer"]
        qa_ids = [item["qa_id"]] + list(item.get("extra_qa_ids") or [])
        context = context_for(qa_ids)
        base = len(examples)
        variants = [
            question,
            noisy_variant(question, base),
            "Please answer: %s" % question,
        ]
        for variant in dict.fromkeys(variants):
            examples.append(
                example(
                    system,
                    context,
                    variant,
                    answer,
                    answer_format,
                    emotion="neutral",
                    metadata={"kind": "hinglish", "variant": variant},
                )
            )

    for index, (question, context) in enumerate(REFUSAL_EXAMPLES):
        variants = [
            question,
            spoken_variant(question, index),
            noisy_variant(question, index),
        ]
        for variant in dict.fromkeys(variants):
            examples.append(
                example(
                    system,
                    context,
                    variant,
                    HELPLINE_ANSWER,
                    answer_format,
                    emotion="empathetic",
                    metadata={"kind": "refusal", "variant": variant},
                )
            )

    return examples


def split_examples(examples: list[dict], train_ratio: float):
    train: list[dict] = []
    val: list[dict] = []

    for item in examples:
        digest = hashlib.sha1(
            json.dumps(item["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF

        if bucket < train_ratio:
            train.append(item)
        else:
            val.append(item)

    # Guarantee a non-empty validation split even for a tiny corpus.
    if not val and len(train) > 1:
        val.append(train.pop())

    return train, val


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    chunks = load_chunks(args.source)
    if not chunks:
        print("No knowledge chunks found under %s" % args.source)
        return 1

    # Optional voice corpus gives us the TTS-normalised answer and the
    # generated ASR aliases for each record. The dataset still builds when
    # it is absent (the script is independent of the enrichment step).
    voice_map = load_voice_alias_map(
        PROJECT_ROOT / "knowledge_base" / "greeto_voice_kb.jsonl"
    )

    examples = build_examples(
        chunks,
        args.max_context_chars,
        args.answer_format,
        voice_map=voice_map,
    )
    if not examples:
        print("No Q&A records could be converted into training examples.")
        return 1

    train, val = split_examples(examples, args.train_ratio)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    train_path = output_dir / "qwen25_train.jsonl"
    val_path = output_dir / "qwen25_val.jsonl"

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    kinds: dict[str, int] = {}
    for row in examples:
        kind = row["metadata"].get("kind", "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1

    print("Corpus: %d chunks (%d Q&A)" % (len(chunks), sum(1 for c in chunks if c.kind == "qa")))
    print("Examples: %d %s" % (len(examples), kinds))
    print("Train: %d -> %s" % (len(train), train_path))
    print("Val:   %d -> %s" % (len(val), val_path))
    print("\nSample:")
    print(json.dumps(train[0], ensure_ascii=False, indent=2)[:1200])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
