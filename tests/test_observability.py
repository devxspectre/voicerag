import json
from pathlib import Path

import pytest

from core.chunker import Chunk, ChunkPosition
from core.pipeline import RagPipeline
from observability import ObservabilityRecorder


class StaticEmbedder:
    def embed_text(self, _text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_texts(self, texts) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]


def make_chunk(chunk_id: str, parent_id: str | None, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        tenant_id="tenant-a",
        source_file="docs/source.pdf",
        document_id="doc-1",
        position=ChunkPosition(page=1, chunk_index=0),
        parent_id=parent_id,
        text=text,
    )


def test_observability_writes_jsonl_records_for_pipeline_query(tmp_path: Path) -> None:
    output_path = tmp_path / "traces.jsonl"
    pipeline = RagPipeline(
        embedder=StaticEmbedder(),
        answer_generator=lambda _prompt: "The answer is rice.",
        observability=ObservabilityRecorder(output_path),
    )
    pipeline.index_chunks(
        [make_chunk("parent-1", None, "Rice context.")],
        [make_chunk("child-1", "parent-1", "Rice context.")],
    )

    result = pipeline.query("tenant-a", "What crop?", top_k=1)

    assert result.answer == "The answer is rice."
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    names = {record["name"] for record in records}
    assert {"rag.query", "rag.retrieve", "rag.prompt", "rag.generate"} <= names
    query_record = next(record for record in records if record["name"] == "rag.query")
    assert query_record["status"] == "ok"
    assert query_record["attributes"]["citation.child_ids"] == ["child-1"]
    assert query_record["attributes"]["retrieval.scores.combined"]


def test_observability_captures_errors_without_swallowing_them(tmp_path: Path) -> None:
    output_path = tmp_path / "traces.jsonl"
    pipeline = RagPipeline(
        embedder=StaticEmbedder(),
        observability=ObservabilityRecorder(output_path),
    )
    pipeline.index_chunks(
        [make_chunk("parent-1", None, "Rice context.")],
        [make_chunk("child-1", "parent-1", "Rice context.")],
    )

    with pytest.raises(ValueError, match="top_k"):
        pipeline.query("tenant-a", "What crop?", top_k=0)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    error_records = [record for record in records if record["status"] == "error"]
    assert {record["name"] for record in error_records} == {"rag.query", "rag.retrieve"}
    assert error_records[0]["error"]["type"] == "ValueError"
