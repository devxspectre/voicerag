from __future__ import annotations

from collections.abc import Callable, Iterable

from storage.vector_store import HybridRetrievalResult, InMemoryHybridVectorStore


EmbeddingFunction = Callable[[str], Iterable[float]]


class HybridRetriever:
    def __init__(
        self,
        store: InMemoryHybridVectorStore,
        embed_query: EmbeddingFunction,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        max_children_per_parent: int = 2,
    ) -> None:
        self.store = store
        self.embed_query = embed_query
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.max_children_per_parent = max_children_per_parent

    def retrieve(
        self,
        tenant_id: str,
        question: str,
        top_k: int = 5,
    ) -> HybridRetrievalResult:
        return self.store.search(
            tenant_id=tenant_id,
            query=question,
            query_embedding=self.embed_query(question),
            top_k=top_k,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
            max_children_per_parent=self.max_children_per_parent,
        )
