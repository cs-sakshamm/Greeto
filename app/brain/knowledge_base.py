"""Knowledge base with hybrid lexical + vector retrieval.

The retriever keeps the original IDF/coverage scorer (which is excellent
at exact admission identifiers) and fuses it with a dense vector ranking
using Reciprocal Rank Fusion (RRF). Both rankings operate over the same
:class:`~app.brain.vector_store.KBChunk` records, so every result can
carry a source, section and confidence back to the conversation layer.

Design goals:

* exact identifiers such as ``1800-180-0840``, ``2,56,500``,
  ``CNRB0018556`` and ``201310`` rank cleanly;
* retrieval stays in the 15-25 ms CPU budget for a few hundred chunks;
* a missing/locked vector index degrades to lexical-only instead of
  breaking the kiosk;
* the returned context fits inside ``prompt.max_context_chars``.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.brain.query_normalizer import normalize_query
from app.brain.vector_store import (
    KBChunk,
    VectorStore,
    build_embedder,
    build_index,
    index_is_current,
    load_chunks,
    strip_citations,
    warm_up_embedder,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Backwards-compatible private name used by older tests/scripts.
_strip_citations = strip_citations

# Query words that carry no admission meaning. Without this, "the" and
# "what" appear in hundreds of chunks and drown the real subject word.
_QUERY_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "am", "be", "been",
    "what", "which", "who", "whom", "whose", "how", "why", "when",
    "where", "do", "does", "did", "can", "could", "should", "would",
    "will", "shall", "i", "me", "my", "we", "our", "you", "your",
    "of", "to", "for", "in", "on", "at", "by", "with", "and", "or",
    "about", "please", "tell", "give", "there", "it", "this", "that",
    "these", "those", "have", "has", "had", "from", "as", "if", "any",
    "much", "many", "know", "want", "need", "would", "like",
    "number", "numbers", "detail", "details", "information", "info",
    "college", "institute", "institution", "greeto",
    "process", "procedure", "policy", "policies", "rule", "rules",
    "system", "work", "works", "happen", "happens", "exactly",
    "currently", "actually", "kindly", "also",
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9,./-]*", re.IGNORECASE)
_DIGIT_RUN_PATTERN = re.compile(r"\d[\d,\-\s]{3,}\d")

_DEFAULTS = {
    "enabled": True,
    "index_dir": "data/vector_index",
    "auto_build": True,
    "sources": ["knowledge_base"],
    "fallback_sources": ["app/knowledge"],
    "embedding": {
        "backend": "auto",
        "model": "BAAI/bge-small-en-v1.5",
        "hashing_dim": 384,
    },
    "retrieval": {
        "top_k": 3,
        "dense_top_k": 8,
        "rrf_k": 60,
        "lexical_weight": 1.0,
        "dense_weight": 1.0,
        "min_dense_similarity": 0.05,
        "max_chunk_chars": 500,
        "max_context_chars": 1100,
        "min_confidence": 0.30,
        "confidence_gate": True,
        "lexical_anchor": True,
        "lexical_anchor_min_score": 120.0,
    },
}


@dataclass
class RetrievalMatch:
    chunk_id: str
    source: str
    section: str
    qa_id: str | None
    score: float


@dataclass
class RetrievalResult:
    """Context plus the evidence needed to decide whether to answer."""

    query: str
    context: str = ""
    confidence: float = 0.0
    matches: list[RetrievalMatch] = field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return bool(self.context.strip())

    @property
    def low_confidence(self) -> bool:
        return not self.has_evidence

    def as_context(self) -> str:
        return self.context


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class KnowledgeBase:
    """Hybrid lexical/dense admission knowledge retrieval."""

    def __init__(
        self,
        config=None,
        knowledge_folder=None,
        index_dir=None,
        embedder=None,
    ) -> None:
        self.config = config

        self.rag = self._load_rag_config(config)

        self.min_confidence = float(
            self._rag_value("retrieval", "min_confidence", default=0.30)
        )
        self.confidence_gate = bool(
            self._rag_value("retrieval", "confidence_gate", default=True)
        )

        self.knowledge_folder = self._resolve_knowledge_folder(knowledge_folder)
        self.index_dir = self._resolve_index_dir(index_dir)

        self.chunks: list[KBChunk] = load_chunks(self.knowledge_folder) if self.knowledge_folder else []

        # Compatibility view for code that inspected the old .documents list.
        self.documents = [
            {
                "source": chunk.source,
                "section": chunk.section,
                "content": chunk.text,
                "qa_id": chunk.qa_id,
                "kind": chunk.kind,
                "kb_id": chunk.kb_id,
                "category": chunk.category,
                "effective_session": chunk.effective_session,
                "volatility": chunk.volatility,
                "search_aliases": list(chunk.search_aliases),
            }
            for chunk in self.chunks
        ]

        # Aliases are part of the lexical haystack, but never part of the
        # chunk text sent to the model. This is what lets a noisy ASR query
        # ("hostile fees") reach the atomic hostel/fee records.
        self._searchable: list[str] = [
            (
                chunk.section
                + " "
                + chunk.text
                + " "
                + " ".join(chunk.search_aliases)
            ).casefold()
            for chunk in self.chunks
        ]
        self._section_text: list[str] = [
            chunk.section.casefold() for chunk in self.chunks
        ]
        self._token_sets: list[set[str]] = [
            self._tokenize(chunk.text) for chunk in self.chunks
        ]
        self._digit_text: list[str] = [
            self._digit_signature(chunk.text) for chunk in self.chunks
        ]

        self.vector_store: VectorStore | None = None
        self.embedder = embedder
        self.index_embedder = None

        if self._rag_value("enabled", default=True):
            self._init_vector_store(embedder)

        logger.info(
            "KnowledgeBase loaded %d chunks (vector=%s, folder=%s)",
            len(self.chunks),
            "yes" if self.vector_store is not None else "no",
            self.knowledge_folder,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rag_config(config) -> dict:
        if config is None:
            try:
                from app.core.config_manager import ConfigManager

                manager = ConfigManager()
                manager.load()
                config = manager
            except Exception as error:  # noqa: BLE001 - config is optional
                logger.debug("RAG configuration unavailable: %s", error)
                return dict(_DEFAULTS)

        if isinstance(config, dict):
            raw = config.get("rag", config)
        else:
            try:
                raw = config.get("rag") or {}
            except (AttributeError, TypeError):
                raw = {}

        return _merge(_DEFAULTS, raw if isinstance(raw, dict) else {})

    def _rag_value(self, *keys, default=None):
        value = self.rag
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return default if value is None else value

    def _resolve_knowledge_folder(self, explicit) -> Path | None:
        candidates: list[Path] = []

        if explicit:
            candidates.append(Path(explicit))

        for name in self._rag_value("sources", default=[]) or []:
            candidates.append(Path(name))

        for path in candidates:
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                return path

        for name in self._rag_value("fallback_sources", default=[]) or []:
            path = Path(name)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                logger.info("Primary knowledge source missing; using %s", path)
                return path

        logger.warning("No knowledge folder found; retrieval will be empty.")
        return None

    def _resolve_index_dir(self, explicit) -> Path:
        raw = explicit or self._rag_value("index_dir", default="data/vector_index")
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def _init_vector_store(self, explicit_embedder=None) -> None:
        if not self.chunks:
            return

        auto_build = bool(self._rag_value("auto_build", default=True))

        if index_is_current(self.index_dir, self.chunks):
            store = VectorStore.load(self.index_dir)
            if store is not None:
                self.vector_store = store
                self.index_embedder = store.embedder
                return

        if not auto_build:
            logger.info("Vector index missing/stale and auto_build is off.")
            return

        embedding = self._rag_value("embedding", default={}) or {}

        try:
            embedder = explicit_embedder or build_embedder(
                backend=embedding.get("backend", "auto"),
                model_name=embedding.get("model", "BAAI/bge-small-en-v1.5"),
                hashing_dim=int(embedding.get("hashing_dim", 384)),
            )

            self.vector_store = build_index(self.chunks, self.index_dir, embedder)
            self.index_embedder = embedder

            logger.info(
                "Built vector index (%d chunks, embedder=%s) at %s",
                len(self.chunks),
                getattr(embedder, "name", "?"),
                self.index_dir,
            )

        except Exception as error:  # noqa: BLE001 - lexical fallback is valid
            logger.warning(
                "Vector index build failed (%s); using lexical retrieval only.",
                error,
            )
            self.vector_store = None

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens: set[str] = set()
        for raw in _TOKEN_PATTERN.findall((text or "").casefold()):
            token = raw.strip(".,/-")
            if len(token) > 1:
                tokens.add(token)
            # Hyphenated institutional compounds ("Pre-Enrollment",
            # "Anti-Ragging", "branch-wise") should also match their parts.
            if "-" in token:
                for part in token.split("-"):
                    part = part.strip(".,/-")
                    if len(part) > 1:
                        tokens.add(part)
        return tokens

    @staticmethod
    def _digit_signature(text: str) -> str:
        """Compact digits so 2,56,500 and 256500 compare equal."""

        return re.sub(r"\D", "", text or "")

    def _query_terms(self, query: str) -> set[str]:
        terms: set[str] = set()
        for raw in _TOKEN_PATTERN.findall((query or "").casefold()):
            token = raw.strip(".,/-")
            if len(token) > 2 and token not in _QUERY_STOPWORDS:
                terms.add(token)
            if "-" in token:
                for part in token.split("-"):
                    part = part.strip(".,/-")
                    if len(part) > 2 and part not in _QUERY_STOPWORDS:
                        terms.add(part)
        return terms

    def _query_digit_runs(self, query: str) -> list[str]:
        runs = []
        for raw in _DIGIT_RUN_PATTERN.findall(query or ""):
            compact = re.sub(r"\D", "", raw)
            if len(compact) >= 4:
                runs.append(compact)
        return runs

    # ------------------------------------------------------------------
    # Lexical ranking
    # ------------------------------------------------------------------

    def _lexical_rank(self, query: str, words: set[str]) -> list[tuple[float, int]]:
        if not words or not self.chunks:
            return []

        total_docs = max(1, len(self.chunks))

        document_frequency: dict[str, int] = {}
        for word in words:
            document_frequency[word] = sum(
                1 for searchable in self._searchable if word in searchable
            )

        weights = {
            word: (math.log((total_docs + 1) / (df + 1)) + 1.0)
            for word, df in document_frequency.items()
        }

        present_words = [word for word in words if document_frequency[word] > 0]

        if not present_words:
            return []

        query_lower = " ".join(query.casefold().split())
        digit_runs = self._query_digit_runs(query)

        results: list[tuple[float, int]] = []

        for index, searchable in enumerate(self._searchable):

            # Do not hard-gate on one rare query word. Alias expansion can
            # introduce a token ("abdul") that only an unrelated overview
            # contains, which would hide the fee record that matches the
            # other eight query terms. IDF weighting below handles noise.
            if not any(word in searchable for word in present_words):
                continue

            section = self._section_text[index]
            score = 0.0

            if query_lower in searchable:
                score += 60.0

            # Coverage matters more than repetition.
            score += 25.0 * sum(1 for word in words if word in searchable)

            for word in words:
                weight = weights[word]

                if word in searchable:
                    score += 10.0 * weight

                if word in section:
                    score += 8.0 * weight

                if word.endswith("s"):
                    if word[:-1] in searchable:
                        score += 5.0 * weight
                elif word + "s" in searchable:
                    score += 5.0 * weight

            # Exact identifiers are the one thing a small model must never
            # lose: phone numbers, IFSC codes and fee amounts.
            digits = self._digit_text[index]
            for run in digit_runs:
                if run and run in digits:
                    score += 50.0

            # Atomic Q&A answers are the quotable ground truth. A long
            # narrative overview often repeats every branch name and would
            # otherwise outrank the one record that actually answers.
            if self.chunks[index].kind == "qa":
                score += 12.0

            if score:
                results.append((score, index))

        results.sort(key=lambda item: item[0], reverse=True)

        return results

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str) -> str:
        """Return bounded, high-signal context for the local model."""

        return self.search_with_confidence(query).context

    def warm_up(self) -> float:
        """Prime the embedder and retrieval path at startup.

        Returns the warm-up wall time in milliseconds. Called alongside the
        Piper TTS warm-up so the first live visitor query pays no cold
        model/numpy allocation penalty.
        """

        started = time.perf_counter()

        if self.index_embedder is not None:
            warm_up_embedder(self.index_embedder)
        elif self.embedder is not None:
            warm_up_embedder(self.embedder)

        if self.chunks:
            self.search_with_confidence("admission helpline fee placement")

        elapsed = (time.perf_counter() - started) * 1000.0
        logger.info("KnowledgeBase warm-up completed in %.1fms", elapsed)
        return elapsed

    def search_with_confidence(self, query: str) -> RetrievalResult:
        """Return context, confidence and the top matching chunks.

        ``query`` is the visitor's raw transcript. Only the retrieval copy
        is alias-expanded; the result keeps the original query so the
        conversation layer can still send the raw wording to the LLM.
        """

        result = RetrievalResult(query=query)

        if not query or not query.strip() or not self.chunks:
            return result

        retrieval_query = normalize_query(query)
        words = self._query_terms(retrieval_query)

        lexical = self._lexical_rank(retrieval_query, words)

        dense: list[tuple[float, KBChunk]] = []
        if self.vector_store is not None and words:
            try:
                dense = self.vector_store.search(
                    retrieval_query,
                    top_k=int(self._rag_value("retrieval", "dense_top_k", default=8)),
                    min_similarity=float(
                        self._rag_value(
                            "retrieval", "min_dense_similarity", default=0.05
                        )
                    ),
                )
            except Exception as error:  # noqa: BLE001 - lexical fallback
                logger.warning("Dense retrieval failed: %s", error)

        if not lexical and not dense:
            return result

        rrf_k = float(self._rag_value("retrieval", "rrf_k", default=60))
        lexical_weight = float(
            self._rag_value("retrieval", "lexical_weight", default=1.0)
        )
        dense_weight = float(
            self._rag_value("retrieval", "dense_weight", default=1.0)
        )

        fused_scores: dict[int, float] = {}
        lexical_scores: dict[int, float] = {}

        for rank, (score, index) in enumerate(lexical[: max(20, len(words) * 4)], start=1):
            fused_scores[index] = fused_scores.get(index, 0.0) + (
                lexical_weight / (rrf_k + rank)
            )
            lexical_scores[index] = score

        for rank, (_, chunk) in enumerate(dense, start=1):
            try:
                index = self.chunks.index(chunk)
            except ValueError:
                continue
            fused_scores[index] = fused_scores.get(index, 0.0) + (
                dense_weight / (rrf_k + rank)
            )

        # A hashed/embedded ranker can prefer a long generic overview that
        # merely mentions the query words over the one atomic Q&A that
        # answers them. When the lexical scorer has a very strong exact
        # match ("documents required for B.Tech admission"), promote that
        # chunk to the front of the fused list. Dense evidence still fills
        # the remaining context slots.
        anchor_threshold = float(
            self._rag_value(
                "retrieval", "lexical_anchor_min_score", default=120.0
            )
        )

        if (
            lexical
            and bool(self._rag_value("retrieval", "lexical_anchor", default=True))
        ):
            # Promote the strongest *atomic* match, not a narrative overview
            # that merely repeats the query vocabulary.
            anchor = next(
                (
                    (score, index)
                    for score, index in lexical
                    if score >= anchor_threshold
                    and self.chunks[index].kind == "qa"
                ),
                None,
            )

            if anchor is not None:
                strongest = anchor[1]
                current_max = max(fused_scores.values(), default=0.0)
                fused_scores[strongest] = max(
                    fused_scores.get(strongest, 0.0),
                    current_max + 1e-6,
                )

        ranked = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_k = max(1, int(self._rag_value("retrieval", "top_k", default=3)))
        selected = ranked[:top_k]

        # Atomic Q&A blocks are the quotable evidence and are usually much
        # shorter than a narrative table. Put them first so the bounded
        # context budget cannot be consumed by a generic overview before
        # the actual answer is included.
        selected = sorted(
            selected,
            key=lambda item: (
                0 if self.chunks[item[0]].kind == "qa" else 1,
                -item[1],
            ),
        )

        # Keep the context inside the domain of the best atomic answer.
        # Mixing in a third chunk from a different section both wastes the
        # 1100-character budget and can pull an unrelated section title into
        # the prompt. If the focus filter would empty the list, keep all.
        qa_sections = [
            self.chunks[index].section
            for index, _ in selected
            if self.chunks[index].kind == "qa"
        ]
        if qa_sections:
            focus = qa_sections[0]
            focused = [
                (index, score)
                for index, score in selected
                if self.chunks[index].section == focus
            ]
            # Only narrow the context when the focus section contributes
            # more than one chunk. A single top hit must not hide an equally
            # strong atomic answer from another domain ("how many seats").
            if len(focused) >= 2:
                selected = focused

        if not selected:
            return result

        # Confidence: query-word coverage in the best chunk plus the best
        # dense similarity and lexical score. It is deliberately simple so
        # it can be explained in a LATENCY/retrieval log line.
        top_index = selected[0][0]
        top_tokens = self._token_sets[top_index]
        coverage = (
            len(words & top_tokens) / len(words) if words else 0.0
        )
        dense_top = float(dense[0][0]) if dense else 0.0
        lexical_top = float(lexical_scores.get(top_index, 0.0))

        confidence = (
            0.45 * coverage
            + 0.35 * max(0.0, min(1.0, dense_top))
            + 0.20 * max(0.0, min(1.0, lexical_top / 80.0))
        )

        result.confidence = round(max(0.0, min(1.0, confidence)), 4)
        result.matches = [
            RetrievalMatch(
                chunk_id=self.chunks[index].chunk_id,
                source=self.chunks[index].source,
                section=self.chunks[index].section,
                qa_id=self.chunks[index].qa_id,
                score=round(score, 6),
            )
            for index, score in selected
        ]

        max_chunk_chars = max(
            120,
            int(self._rag_value("retrieval", "max_chunk_chars", default=500)),
        )
        max_context_chars = max(
            240,
            int(self._rag_value("retrieval", "max_context_chars", default=1100)),
        )

        rarest = [word for word in words if word in top_tokens]
        rarest.sort(
            key=lambda word: (
                sum(1 for searchable in self._searchable if word in searchable),
                word,
            )
        )

        blocks: list[str] = []
        total = 0

        for index, _ in selected:

            chunk = self.chunks[index]

            excerpt = self._best_excerpt(
                chunk.text,
                rarest,
                max_chars=max_chunk_chars,
            )

            block = "Source: %s\nSection: %s\n\n%s" % (
                chunk.source,
                chunk.section,
                excerpt,
            )

            remaining = max_context_chars - total
            if remaining <= 0:
                break

            if len(block) > remaining:
                block = block[:remaining].rstrip() + "…"

            blocks.append(block)
            total += len(block) + 2

        result.context = "\n\n".join(blocks)

        return result

    def is_confident(self, result: RetrievalResult) -> bool:
        """Return True when the evidence is strong enough to answer."""

        if not self.confidence_gate:
            return result.has_evidence

        return result.has_evidence and result.confidence >= self.min_confidence

    # ------------------------------------------------------------------
    # Excerpts
    # ------------------------------------------------------------------

    @staticmethod
    def _best_excerpt(
        content: str,
        words: list[str],
        max_chars: int = 430,
    ) -> str:
        """Return a compact window around the most specific match."""

        if len(content) <= max_chars:
            return content.strip()

        lowered = content.casefold()
        position = -1

        for word in words:
            position = lowered.find(word.casefold())
            if position != -1:
                break

        if position == -1:
            return content[:max_chars].strip()

        start = max(0, position - 140)
        end = min(len(content), start + max_chars)
        start = max(0, end - max_chars)

        # Do not begin or end mid-word; a spoken excerpt that starts
        # "orate Resource Center" is confusing in the prompt.
        if start > 0:
            space = content.find(" ", start, max(start + 1, end - 40))
            if space != -1:
                start = space + 1
        if end < len(content):
            space = content.rfind(" ", start, end)
            if space > start:
                end = space

        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(content) else ""

        return prefix + content[start:end].strip() + suffix
