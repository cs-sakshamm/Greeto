#!/usr/bin/env python3
"""Build the embedded Greeto knowledge-base index.

Usage:

    .venv/Scripts/python.exe scripts/build_vector_index.py
    .venv/Scripts/python.exe scripts/build_vector_index.py --backend hashing
    .venv/Scripts/python.exe scripts/build_vector_index.py --query "MBA fee"

The index is written to ``data/vector_index/`` (gitignored) as
``index.npz`` + ``chunks.jsonl`` + ``meta.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.brain.vector_store import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    build_embedder,
    build_index,
    load_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="knowledge_base",
        help="Knowledge file or directory (default: knowledge_base).",
    )
    parser.add_argument(
        "--index",
        default="data/vector_index",
        help="Output directory (default: data/vector_index).",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "sentence-transformers", "hashing"),
        help="Embedding backend (default: auto).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="sentence-transformers model name.",
    )
    parser.add_argument(
        "--hashing-dim",
        type=int,
        default=384,
        help="Dimension of the hashed fallback embedder.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Run a sample query after building (repeatable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    started = time.perf_counter()
    chunks = load_chunks(args.source)
    if not chunks:
        print("No knowledge chunks found under %s" % args.source)
        return 1

    qa_count = sum(1 for chunk in chunks if chunk.kind == "qa")
    print(
        "Loaded %d chunks (%d atomic Q&A, %d narrative) in %.1f ms"
        % (
            len(chunks),
            qa_count,
            len(chunks) - qa_count,
            (time.perf_counter() - started) * 1000.0,
        )
    )

    embedder = build_embedder(
        backend=args.backend,
        model_name=args.model,
        hashing_dim=args.hashing_dim,
    )

    print("Embedding with %s (dim=%d)..." % (embedder.name, embedder.dim))

    build_started = time.perf_counter()
    store = build_index(chunks, args.index, embedder)
    build_ms = (time.perf_counter() - build_started) * 1000.0

    print(
        "Index written to %s (%d vectors, %.1f ms)"
        % (args.index, len(store), build_ms)
    )

    queries = args.query or [
        "What is the admission helpline number?",
        "What is the total fee for the MBA program?",
        "How long does a refund take?",
    ]

    for query in queries:
        query_started = time.perf_counter()
        hits = store.search(query, top_k=3)
        elapsed = (time.perf_counter() - query_started) * 1000.0

        print('\nQuery: "%s" (%.2f ms)' % (query, elapsed))
        for score, chunk in hits:
            preview = " ".join(chunk.text.split())[:120]
            print("  %.4f  %-10s %s" % (score, chunk.qa_id or chunk.kind, preview))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
