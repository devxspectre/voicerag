from core.chunker import Chunk, ChunkPosition
from core.retriever import HybridRetriever
from storage.vector_store import InMemoryHybridVectorStore, cosine_similarity


def make_chunk(
    chunk_id: str,
    tenant_id: str = "tenant-a",
    parent_id: str | None = None,
    text: str = "sample text",
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        tenant_id=tenant_id,
        source_file="docs/source.pdf",
        document_id="doc-1",
        position=ChunkPosition(page=1, chunk_index=chunk_index),
        parent_id=parent_id,
        text=text,
    )


def build_store() -> InMemoryHybridVectorStore:
    store = InMemoryHybridVectorStore()
    parents = [
        make_chunk("parent-1", text="Parent context about rice fertilizer rates."),
        make_chunk("parent-2", text="Parent context about invoice number ZXQ-7781."),
        make_chunk("parent-3", text="Parent context about unrelated crop rotation."),
        make_chunk(
            "other-parent",
            tenant_id="tenant-b",
            text="Other tenant private context.",
        ),
    ]
    store.add_parent_chunks(parents)
    store.add_child_chunks(
        [
            (
                make_chunk(
                    "child-1",
                    parent_id="parent-1",
                    text="Rice nitrogen fertilizer recommendation.",
                    chunk_index=0,
                ),
                [1.0, 0.0],
            ),
            (
                make_chunk(
                    "child-2",
                    parent_id="parent-1",
                    text="Rice phosphorus fertilizer recommendation.",
                    chunk_index=1,
                ),
                [0.95, 0.05],
            ),
            (
                make_chunk(
                    "child-3",
                    parent_id="parent-1",
                    text="Rice potassium fertilizer recommendation.",
                    chunk_index=2,
                ),
                [0.9, 0.1],
            ),
            (
                make_chunk(
                    "child-4",
                    parent_id="parent-2",
                    text="The exact invoice identifier is ZXQ-7781.",
                    chunk_index=3,
                ),
                [0.0, 1.0],
            ),
            (
                make_chunk(
                    "child-5",
                    parent_id="parent-3",
                    text="Crop rotation and soil rest period.",
                    chunk_index=4,
                ),
                [0.2, 0.8],
            ),
            (
                make_chunk(
                    "other-child",
                    tenant_id="tenant-b",
                    parent_id="other-parent",
                    text="Rice nitrogen private tenant answer.",
                ),
                [1.0, 0.0],
            ),
        ]
    )
    return store


def test_cosine_similarity_prefers_closer_vector() -> None:
    assert cosine_similarity([1, 0], [1, 0]) > cosine_similarity([1, 0], [0, 1])


def test_hybrid_search_uses_bm25_to_recover_exact_fact() -> None:
    store = build_store()

    result = store.search(
        tenant_id="tenant-a",
        query="What is invoice ZXQ-7781?",
        query_embedding=[1.0, 0.0],
        top_k=3,
    )

    chunk_ids = [match.child.chunk_id for match in result.matches]
    assert "child-4" in chunk_ids
    bm25_match = next(match for match in result.matches if match.child.chunk_id == "child-4")
    assert bm25_match.bm25_score > 0


def test_max_two_children_per_parent_are_returned() -> None:
    store = build_store()

    result = store.search(
        tenant_id="tenant-a",
        query="rice fertilizer recommendation",
        query_embedding=[1.0, 0.0],
        top_k=4,
    )

    parent_1_matches = [
        match for match in result.matches if match.child.parent_id == "parent-1"
    ]
    assert len(parent_1_matches) == 2
    assert len(result.matches) == 4


def test_retrieval_filters_by_tenant() -> None:
    store = build_store()

    result = store.search(
        tenant_id="tenant-b",
        query="rice nitrogen",
        query_embedding=[1.0, 0.0],
        top_k=5,
    )

    assert [match.child.chunk_id for match in result.matches] == ["other-child"]
    assert all(match.child.tenant_id == "tenant-b" for match in result.matches)


def test_child_matches_resolve_to_deduplicated_parent_context() -> None:
    store = build_store()
    retriever = HybridRetriever(store=store, embed_query=lambda _query: [1.0, 0.0])

    result = retriever.retrieve(
        tenant_id="tenant-a",
        question="rice fertilizer recommendation",
        top_k=3,
    )

    assert result.matches[0].child.is_child
    assert all(parent.is_parent for parent in result.parents)
    assert len({parent.chunk_id for parent in result.parents}) == len(result.parents)
    assert "Parent context about rice fertilizer rates." in result.context_text
    assert "chunk_id=parent-1" in result.context_text
    assert result.matches[0].child.source_file == "docs/source.pdf"
    assert result.matches[0].child.position.page == 1
