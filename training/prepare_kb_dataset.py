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
from app.brain.vector_store import KBChunk, load_chunks  # noqa: E402

HELPLINE_ANSWER = (
    "I couldn't find that information in the college database. "
    "Please contact the admission office at 1800-180-0840."
)

# Deliberately unanswerable from the static corpus. Each entry is
# (question, nearby context that proves the number is *not* published).
REFUSAL_EXAMPLES = [
    (
        "What is the exact B.Tech tuition fee for the current session?",
        "The official 2026 B.Tech brochure and fee webpage do not publish "
        "itemized monetary B.Tech tuition charges; tuition is regulated by "
        "the Uttar Pradesh State Fee Fixation Committee.",
    ),
    (
        "How many B.Tech CSE seats are still vacant right now?",
        "Sanctioned intake for B.Tech CSE is 750 seats under the 2026-27 "
        "Mandatory Disclosure. Live vacancy changes every counseling round.",
    ),
    (
        "What was the closing JEE Main cutoff in the latest UPTAC round?",
        "Official cutoffs change every round, and the college does not "
        "publish a static rank table.",
    ),
    (
        "What is the hostel fee per year?",
        "The official 2026 brochure does not publish itemized hostel charges. "
        "Hostel refund rules are published separately.",
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
        "direct admission when AKTU eligibility is met and seats are available.",
    ),
    (
        "What is today's live seat availability for MBA?",
        "The MBA sanctioned intake is 60 seats. Live vacancy is not published "
        "in the static knowledge base.",
    ),
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


def build_examples(chunks: list[KBChunk], max_context_chars: int, answer_format: str):
    qa_chunks = [chunk for chunk in chunks if chunk.kind == "qa" and chunk.qa_id]
    by_section: dict[str, list[KBChunk]] = {}
    for chunk in chunks:
        by_section.setdefault(chunk.section, []).append(chunk)

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

        context = build_context(chunk, by_section, max_context_chars)
        metadata = {"qa_id": chunk.qa_id, "section": chunk.section, "kind": "grounded"}

        variants = [
            question,
            spoken_variant(question, len(examples)),
            noisy_variant(question, len(examples)),
        ]

        for variant in dict.fromkeys(variants):
            examples.append(
                example(
                    system,
                    context,
                    variant,
                    answer,
                    answer_format,
                    emotion=pick_emotion(answer),
                    metadata=dict(metadata, variant=variant),
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

    examples = build_examples(chunks, args.max_context_chars, args.answer_format)
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

    grounded = sum(1 for row in examples if row["metadata"].get("kind") == "grounded")
    refusals = len(examples) - grounded

    print("Corpus: %d chunks (%d Q&A)" % (len(chunks), sum(1 for c in chunks if c.kind == "qa")))
    print("Examples: %d (%d grounded, %d refusal)" % (len(examples), grounded, refusals))
    print("Train: %d -> %s" % (len(train), train_path))
    print("Val:   %d -> %s" % (len(val), val_path))
    print("\nSample:")
    print(json.dumps(train[0], ensure_ascii=False, indent=2)[:1200])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
