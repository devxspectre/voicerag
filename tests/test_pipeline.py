from core.chunker import Chunk, ChunkPosition, HierarchicalChunker, PageText
from core.pipeline import RagPipeline


class RecordingEmbedder:
    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        if "invoice" in text.lower() or "zxq" in text.lower():
            return [0.0, 1.0]
        return [1.0, 0.0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [self._embedding_for_text(text) for text in texts]

    def _embedding_for_text(self, text: str) -> list[float]:
        if "invoice" in text.lower() or "zxq" in text.lower():
            return [0.0, 1.0]
        return [1.0, 0.0]


def make_chunk(
    chunk_id: str,
    parent_id: str | None,
    text: str,
    chunk_index: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        tenant_id="tenant-a",
        source_file="docs/source.pdf",
        document_id="doc-1",
        position=ChunkPosition(page=2, chunk_index=chunk_index),
        parent_id=parent_id,
        text=text,
    )


def test_pipeline_embeds_children_and_returns_index_counts() -> None:
    embedder = RecordingEmbedder()
    pipeline = RagPipeline(embedder=embedder)
    parent = make_chunk(
        "parent-1",
        parent_id=None,
        text="Large parent context that should go to the LLM.",
        chunk_index=0,
    )
    children = [
        make_chunk("child-1", "parent-1", "small searchable child", 0),
        make_chunk("child-2", "parent-1", "invoice ZXQ-7781", 1),
    ]

    result = pipeline.index_chunks([parent], children)

    assert result.parent_count == 1
    assert result.child_count == 2
    assert embedder.embedded_texts == [
        "small searchable child",
        "invoice ZXQ-7781",
    ]


def test_pipeline_query_sends_parent_context_and_child_citations() -> None:
    answers: list[str] = []

    def answer_generator(prompt: str) -> str:
        answers.append(prompt)
        return "The invoice identifier is ZXQ-7781."

    pipeline = RagPipeline(
        embedder=RecordingEmbedder(),
        answer_generator=answer_generator,
    )
    parent = make_chunk(
        "parent-1",
        parent_id=None,
        text="Full parent context: The invoice identifier is ZXQ-7781.",
        chunk_index=0,
    )
    children = [
        make_chunk("child-1", "parent-1", "invoice ZXQ-7781", 0),
    ]
    pipeline.index_chunks([parent], children)

    result = pipeline.query(
        tenant_id="tenant-a",
        question="What is the invoice identifier?",
        top_k=1,
    )

    assert result.answer == "The invoice identifier is ZXQ-7781."
    assert result.context == result.retrieval.context_text
    assert "Full parent context" in result.prompt
    assert "I don't see that in the document." in result.prompt
    assert "Keep the answer short" in result.prompt
    assert "child_id=child-1" in result.prompt
    assert "parent_id=parent-1" in result.prompt
    assert result.citations[0].source_file == "docs/source.pdf"
    assert result.citations[0].position["page"] == 2
    assert answers == [result.prompt]


def test_pipeline_returns_none_answer_without_generator() -> None:
    pipeline = RagPipeline(embedder=RecordingEmbedder())
    parent = make_chunk("parent-1", None, "Parent context.", 0)
    child = make_chunk("child-1", "parent-1", "Child context.", 0)
    pipeline.index_chunks([parent], [child])

    result = pipeline.query(
        tenant_id="tenant-a",
        question="What context exists?",
        top_k=1,
    )

    assert result.answer is None
    assert "Parent context." in result.context


def test_pipeline_indexes_page_text_with_real_chunker() -> None:
    pipeline = RagPipeline(embedder=RecordingEmbedder())

    indexed = pipeline.index_pages(
        pages=[
            PageText(
                page=1,
                text=(
                    "1.1 Invoice Section\n"
                    "The invoice identifier is ZXQ-7781. "
                    "This section carries the source value."
                ),
            )
        ],
        tenant_id="tenant-a",
        source_file="docs/source.pdf",
        document_id="doc-1",
        chunker=HierarchicalChunker(
            parent_token_limit=100,
            child_token_limit=12,
            child_overlap_tokens=2,
        ),
    )
    result = pipeline.query(
        tenant_id="tenant-a",
        question="What is the invoice identifier?",
        top_k=2,
    )

    assert indexed.parent_count == 1
    assert indexed.child_count >= 1
    assert "The invoice identifier is ZXQ-7781" in result.context
    assert result.citations[0].parent_id
    assert result.citations[0].position["section"] == "1.1 Invoice Section"
