from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from rank_bm25 import BM25Okapi

from core.chunker import Chunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class RetrievedChunk:
    child: Chunk
    parent: Chunk
    score: float
    dense_score: float
    bm25_score: float


@dataclass(frozen=True)
class HybridRetrievalResult:
    matches: list[RetrievedChunk]
    parents: list[Chunk]

    @property
    def context_text(self) -> str:
        sections = []
        for parent in self.parents:
            position = parent.position.as_payload()
            label = (
                f"source={parent.source_file} "
                f"document_id={parent.document_id} "
                f"chunk_id={parent.chunk_id} "
                f"position={position}"
            )
            sections.append(f"[{label}]\n{parent.text}")
        return "\n\n".join(sections)


def tokenize_for_bm25(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values):
        raise ValueError("Cosine similarity requires vectors with the same dimension")

    dot = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum == minimum:
        value = 1.0 if maximum > 0 else 0.0
        return {chunk_id: value for chunk_id in scores}

    return {
        chunk_id: (score - minimum) / (maximum - minimum)
        for chunk_id, score in scores.items()
    }


class InMemoryHybridVectorStore:
    def __init__(self) -> None:
        self._parents: dict[str, Chunk] = {}
        self._children: dict[str, Chunk] = {}
        self._child_embeddings: dict[str, list[float]] = {}

    def add_parent_chunks(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            chunk.require_valid()
            if not chunk.is_parent:
                raise ValueError(f"Parent chunk {chunk.chunk_id} cannot have parent_id")
            self._parents[chunk.chunk_id] = chunk

    def add_child_chunks(
        self, chunks_with_embeddings: Iterable[tuple[Chunk, Iterable[float]]]
    ) -> None:
        for chunk, embedding in chunks_with_embeddings:
            chunk.require_valid()
            if not chunk.is_child:
                raise ValueError(f"Child chunk {chunk.chunk_id} must have parent_id")
            if chunk.parent_id not in self._parents:
                raise ValueError(
                    f"Child chunk {chunk.chunk_id} references unknown parent "
                    f"{chunk.parent_id}"
                )
            parent = self._parents[chunk.parent_id]
            if parent.tenant_id != chunk.tenant_id:
                raise ValueError(
                    f"Child chunk {chunk.chunk_id} tenant does not match parent"
                )
            self._children[chunk.chunk_id] = chunk
            self._child_embeddings[chunk.chunk_id] = list(embedding)

    def get_parent(self, chunk_id: str) -> Chunk:
        return self._parents[chunk_id]

    def get_child(self, chunk_id: str) -> Chunk:
        return self._children[chunk_id]

    def search(
        self,
        tenant_id: str,
        query: str,
        query_embedding: Iterable[float],
        top_k: int = 5,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        max_children_per_parent: int = 2,
    ) -> HybridRetrievalResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_children_per_parent < 1:
            raise ValueError("max_children_per_parent must be at least 1")

        candidates = [
            child
            for child in self._children.values()
            if child.tenant_id == tenant_id
        ]
        if not candidates:
            return HybridRetrievalResult(matches=[], parents=[])

        query_embedding_values = list(query_embedding)
        dense_raw = {
            child.chunk_id: cosine_similarity(
                query_embedding_values, self._child_embeddings[child.chunk_id]
            )
            for child in candidates
        }
        dense_scores = normalize_scores(dense_raw)

        tokenized_corpus = [tokenize_for_bm25(child.text) for child in candidates]
        bm25_raw: dict[str, float] = {}
        query_tokens = tokenize_for_bm25(query)
        if query_tokens and any(tokenized_corpus):
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_values = bm25.get_scores(query_tokens)
            bm25_raw = {
                child.chunk_id: float(score)
                for child, score in zip(candidates, bm25_values, strict=True)
            }
        else:
            bm25_raw = {child.chunk_id: 0.0 for child in candidates}
        bm25_scores = normalize_scores(bm25_raw)

        ranked = sorted(
            candidates,
            key=lambda child: (
                dense_weight * dense_scores.get(child.chunk_id, 0.0)
                + bm25_weight * bm25_scores.get(child.chunk_id, 0.0),
                dense_scores.get(child.chunk_id, 0.0),
                bm25_scores.get(child.chunk_id, 0.0),
            ),
            reverse=True,
        )

        parent_counts: dict[str, int] = {}
        matches: list[RetrievedChunk] = []
        for child in ranked:
            parent_id = child.parent_id
            if parent_id is None:
                continue
            if parent_counts.get(parent_id, 0) >= max_children_per_parent:
                continue

            parent = self._parents[parent_id]
            combined_score = (
                dense_weight * dense_scores.get(child.chunk_id, 0.0)
                + bm25_weight * bm25_scores.get(child.chunk_id, 0.0)
            )
            matches.append(
                RetrievedChunk(
                    child=child,
                    parent=parent,
                    score=combined_score,
                    dense_score=dense_scores.get(child.chunk_id, 0.0),
                    bm25_score=bm25_scores.get(child.chunk_id, 0.0),
                )
            )
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
            if len(matches) == top_k:
                break

        parents = _dedupe_parents(matches)
        return HybridRetrievalResult(matches=matches, parents=parents)


def _dedupe_parents(matches: Iterable[RetrievedChunk]) -> list[Chunk]:
    seen: set[str] = set()
    parents: list[Chunk] = []
    for match in matches:
        if match.parent.chunk_id in seen:
            continue
        seen.add(match.parent.chunk_id)
        parents.append(match.parent)
    return parents
