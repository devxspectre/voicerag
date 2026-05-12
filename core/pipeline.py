from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from core.chunker import Chunk, HierarchicalChunker, PageText
from core.embedder import Embedder
from core.retriever import HybridRetriever
from storage.vector_store import (
    HybridRetrievalResult,
    InMemoryHybridVectorStore,
    RetrievedChunk,
)
from observability import ObservabilityRecorder


AnswerGenerator = Callable[[str], str]


@dataclass(frozen=True)
class IndexingResult:
    parent_count: int
    child_count: int


@dataclass(frozen=True)
class Citation:
    child_id: str
    parent_id: str
    source_file: str
    document_id: str
    position: dict[str, Any]
    score: float
    dense_score: float
    bm25_score: float


@dataclass(frozen=True)
class PipelineQueryResult:
    question: str
    answer: str | None
    prompt: str
    context: str
    citations: list[Citation]
    retrieval: HybridRetrievalResult


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: InMemoryHybridVectorStore | None = None,
        answer_generator: AnswerGenerator | None = None,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        max_children_per_parent: int = 2,
        observability: ObservabilityRecorder | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store or InMemoryHybridVectorStore()
        self.answer_generator = answer_generator
        self.observability = observability
        self.retriever = HybridRetriever(
            store=self.store,
            embed_query=self.embedder.embed_text,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            max_children_per_parent=max_children_per_parent,
        )

    def index_chunks(
        self,
        parent_chunks: Iterable[Chunk],
        child_chunks: Iterable[Chunk],
    ) -> IndexingResult:
        parents = list(parent_chunks)
        children = list(child_chunks)

        self.store.add_parent_chunks(parents)
        child_embeddings = self.embedder.embed_texts([chunk.text for chunk in children])
        self.store.add_child_chunks(zip(children, child_embeddings, strict=True))

        return IndexingResult(parent_count=len(parents), child_count=len(children))

    def index_pages(
        self,
        pages: list[PageText],
        tenant_id: str,
        source_file: str,
        document_id: str | None = None,
        chunker: HierarchicalChunker | None = None,
    ) -> IndexingResult:
        chunker = chunker or HierarchicalChunker()
        chunks = chunker.chunk_pages(
            pages=pages,
            tenant_id=tenant_id,
            source_file=source_file,
            document_id=document_id,
        )
        return self.index_chunks(chunks.parents, chunks.children)

    def retrieve(
        self,
        tenant_id: str,
        question: str,
        top_k: int = 5,
    ) -> HybridRetrievalResult:
        return self.retriever.retrieve(
            tenant_id=tenant_id,
            question=question,
            top_k=top_k,
        )

    def query(
        self,
        tenant_id: str,
        question: str,
        top_k: int = 5,
    ) -> PipelineQueryResult:
        if self.observability is None:
            return self._query_unobserved(
                tenant_id=tenant_id,
                question=question,
                top_k=top_k,
            )
        return self._query_observed(
            tenant_id=tenant_id,
            question=question,
            top_k=top_k,
        )

    def _query_unobserved(
        self,
        tenant_id: str,
        question: str,
        top_k: int,
    ) -> PipelineQueryResult:
        retrieval = self.retrieve(
            tenant_id=tenant_id,
            question=question,
            top_k=top_k,
        )
        citations = [_citation_from_match(match) for match in retrieval.matches]
        prompt = build_rag_prompt(question=question, retrieval=retrieval)
        answer = self.answer_generator(prompt) if self.answer_generator else None
        return _query_result(question, answer, prompt, retrieval, citations)

    def _query_observed(
        self,
        tenant_id: str,
        question: str,
        top_k: int,
    ) -> PipelineQueryResult:
        assert self.observability is not None
        root_attrs = {
            "tenant_id": tenant_id,
            "top_k": top_k,
            "question.length": len(question),
            "answer_generator.enabled": self.answer_generator is not None,
        }
        with self.observability.span("rag.query", root_attrs) as query_span:
            trace_id = str(query_span.attributes["trace_id"])
            parent_span_id = str(query_span.attributes["span_id"])
            with self.observability.span(
                "rag.retrieve",
                {"tenant_id": tenant_id, "top_k": top_k, "question.length": len(question)},
                parent_span_id=parent_span_id,
                trace_id=trace_id,
            ) as retrieve_span:
                retrieval = self.retrieve(
                    tenant_id=tenant_id,
                    question=question,
                    top_k=top_k,
                )
                retrieve_span.set_attributes(_retrieval_attributes(retrieval))

            citations = [_citation_from_match(match) for match in retrieval.matches]
            with self.observability.span(
                "rag.prompt",
                _retrieval_attributes(retrieval),
                parent_span_id=parent_span_id,
                trace_id=trace_id,
            ) as prompt_span:
                prompt = build_rag_prompt(question=question, retrieval=retrieval)
                prompt_span.set_attribute("prompt.length", len(prompt))
                prompt_span.set_attribute("context.length", len(retrieval.context_text))

            answer = None
            if self.answer_generator:
                with self.observability.span(
                    "rag.generate",
                    {"prompt.length": len(prompt)},
                    parent_span_id=parent_span_id,
                    trace_id=trace_id,
                ) as generation_span:
                    answer = self.answer_generator(prompt)
                    generation_span.set_attribute("answer.length", len(answer or ""))

            query_span.set_attributes(
                {
                    **_retrieval_attributes(retrieval),
                    "citation.child_ids": [citation.child_id for citation in citations],
                    "citation.parent_ids": [citation.parent_id for citation in citations],
                    "context.length": len(retrieval.context_text),
                    "prompt.length": len(prompt),
                    "answer.length": len(answer or ""),
                }
            )
            return _query_result(question, answer, prompt, retrieval, citations)


def build_rag_prompt(question: str, retrieval: HybridRetrievalResult) -> str:
    citations = "\n".join(
        (
            f"- child_id={match.child.chunk_id}; parent_id={match.parent.chunk_id}; "
            f"source={match.child.source_file}; "
            f"position={match.child.position.as_payload()}; "
            f"score={match.score:.4f}"
        )
        for match in retrieval.matches
    )
    if not citations:
        citations = "- No retrieved child chunks."

    context = retrieval.context_text or "No context was retrieved."
    return (
        "Decide whether the question is about the document/resume/candidate or "
        "is a general conversation question. If it is document-bound, answer "
        "using the parent context below as the source of truth. If the parent "
        "context does not contain the answer, say exactly: \"I don't see that "
        "in the document.\" If it is clearly general, answer normally and do "
        "not force the document context. Keep the answer short: 1-3 sentences "
        "or at most 3 bullets. If this is a voice-style question, answer "
        "conversationally rather than copying resume formatting. Mention "
        "source/page only if it helps disambiguate.\n\n"
        f"Question:\n{question}\n\n"
        f"Parent context:\n{context}\n\n"
        f"Matched child citations:\n{citations}"
    )


def _query_result(
    question: str,
    answer: str | None,
    prompt: str,
    retrieval: HybridRetrievalResult,
    citations: list[Citation],
) -> PipelineQueryResult:
    return PipelineQueryResult(
        question=question,
        answer=answer,
        prompt=prompt,
        context=retrieval.context_text,
        citations=citations,
        retrieval=retrieval,
    )


def _retrieval_attributes(retrieval: HybridRetrievalResult) -> dict[str, Any]:
    return {
        "retrieval.match_count": len(retrieval.matches),
        "retrieval.parent_count": len(retrieval.parents),
        "retrieval.child_ids": [match.child.chunk_id for match in retrieval.matches],
        "retrieval.parent_ids": [match.parent.chunk_id for match in retrieval.matches],
        "retrieval.scores.combined": [match.score for match in retrieval.matches],
        "retrieval.scores.dense": [match.dense_score for match in retrieval.matches],
        "retrieval.scores.bm25": [match.bm25_score for match in retrieval.matches],
    }


def _citation_from_match(match: RetrievedChunk) -> Citation:
    parent_id = match.child.parent_id
    if parent_id is None:
        raise ValueError(f"Retrieved child {match.child.chunk_id} has no parent_id")

    return Citation(
        child_id=match.child.chunk_id,
        parent_id=parent_id,
        source_file=match.child.source_file,
        document_id=match.child.document_id,
        position=match.child.position.as_payload(),
        score=match.score,
        dense_score=match.dense_score,
        bm25_score=match.bm25_score,
    )
