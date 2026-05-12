import json
import sys
from pathlib import Path

from core.chunker import Chunk, ChunkPosition
from core.pipeline import RagPipeline
from evals.chat import answer_once_with_eval, parse_args
from evals.runner import EvalCase, JudgeResult


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


def build_pipeline() -> RagPipeline:
    pipeline = RagPipeline(
        embedder=StaticEmbedder(),
        answer_generator=lambda _prompt: "The answer is ZXQ-7781.",
    )
    pipeline.index_chunks(
        [make_chunk("parent-1", None, "The invoice identifier is ZXQ-7781.")],
        [make_chunk("child-1", "parent-1", "invoice identifier ZXQ-7781")],
    )
    return pipeline


def test_live_eval_prints_verdict_and_records_jsonl(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_judge(question, result) -> JudgeResult:
        assert question == "What is the invoice identifier?"
        assert result.answer == "The answer is ZXQ-7781."
        return JudgeResult(score=0.91, passed=True, reason="Faithful and relevant.")

    answer_once_with_eval(
        pipeline=build_pipeline(),
        tenant_id="tenant-a",
        question="What is the invoice identifier?",
        top_k=1,
        judge=fake_judge,
        cases=[
            EvalCase(
                question="What is the invoice identifier?",
                expected_child_ids=["child-1"],
            )
        ],
    )

    output = capsys.readouterr().out
    assert "Eval> PASS score=0.91 - Faithful and relevant." in output
    assert "Metrics> precision@k=0.50 recall@k=1.00 mrr=1.00 hit_rate=1.00" in output
    record_path = tmp_path / "outputs/evals/live_results.jsonl"
    record = json.loads(record_path.read_text().splitlines()[0])
    assert record["question"] == "What is the invoice identifier?"
    assert record["judge"]["passed"] is True
    assert record["citations"][0]["child_id"] == "child-1"
    assert record["metrics"]["mrr"] == 1.0


def test_live_eval_judge_failure_does_not_hide_answer(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    def broken_judge(_question, _result) -> JudgeResult:
        raise RuntimeError("judge unavailable")

    answer_once_with_eval(
        pipeline=build_pipeline(),
        tenant_id="tenant-a",
        question="What is the invoice identifier?",
        top_k=1,
        judge=broken_judge,
    )

    output = capsys.readouterr().out
    assert "Mirinda> The answer is ZXQ-7781." in output
    assert "Eval> ERROR - judge unavailable" in output
    assert "Metrics> precision@k=n/a recall@k=n/a mrr=n/a hit_rate=n/a" in output
    record = json.loads(
        (tmp_path / "outputs/evals/live_results.jsonl").read_text().splitlines()[0]
    )
    assert record["error"] == "judge unavailable"
    assert record["judge"] is None
    assert record["metrics"] is None


def test_eval_chat_parse_args_accepts_pdf(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["python -m evals.chat", "resume_remote.pdf"])

    args = parse_args()

    assert args.pdf == Path("resume_remote.pdf")
    assert args.top_k == 5
