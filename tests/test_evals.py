import json
import sys
from pathlib import Path

from core.chunker import Chunk, ChunkPosition
from core.pipeline import RagPipeline
from evals.run import parse_args
from evals.runner import (
    EvalCase,
    JudgeResult,
    load_cases,
    run_evals,
    write_eval_outputs,
)


class StaticEmbedder:
    def embed_text(self, text: str) -> list[float]:
        if "invoice" in text.lower() or "zxq" in text.lower():
            return [0.0, 1.0]
        return [1.0, 0.0]

    def embed_texts(self, texts) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


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


def build_pipeline(answer: str = "The invoice identifier is ZXQ-7781.") -> RagPipeline:
    pipeline = RagPipeline(
        embedder=StaticEmbedder(),
        answer_generator=lambda _prompt: answer,
    )
    pipeline.index_chunks(
        [make_chunk("parent-1", None, "The invoice identifier is ZXQ-7781.")],
        [make_chunk("child-1", "parent-1", "invoice identifier ZXQ-7781")],
    )
    return pipeline


def test_deterministic_eval_passes_for_answer_and_citations() -> None:
    summary = run_evals(
        pipeline=build_pipeline(),
        tenant_id="tenant-a",
        cases=[
            EvalCase(
                question="What is the invoice identifier?",
                expected_answer_contains=["ZXQ-7781"],
                expected_child_ids=["child-1"],
                expected_parent_ids=["parent-1"],
                expected_source_file="docs/source.pdf",
            )
        ],
        top_k=1,
    )

    assert summary.total == 1
    assert summary.passed == 1
    assert summary.results[0].checks == {
        "answer_contains": True,
        "child_ids": True,
        "parent_ids": True,
        "source_file": True,
    }


def test_deterministic_eval_fails_for_missing_expected_answer() -> None:
    summary = run_evals(
        pipeline=build_pipeline(answer="I don't see that in the document."),
        tenant_id="tenant-a",
        cases=[
            EvalCase(
                question="What is the invoice identifier?",
                expected_answer_contains=["ZXQ-7781"],
            )
        ],
        top_k=1,
    )

    assert summary.failed == 1
    assert summary.results[0].checks["answer_contains"] is False


def test_eval_runner_writes_predefined_result_and_summary_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    summary = run_evals(
        pipeline=build_pipeline(),
        tenant_id="tenant-a",
        cases=[EvalCase(question="What is the invoice identifier?")],
        top_k=1,
    )

    write_eval_outputs(summary)

    result_path = tmp_path / "outputs/evals/results.jsonl"
    summary_path = tmp_path / "outputs/evals/summary.json"
    assert result_path.exists()
    assert summary_path.exists()
    assert json.loads(result_path.read_text().splitlines()[0])["passed"] is True
    assert json.loads(summary_path.read_text())["total"] == 1


def test_judge_mode_uses_injected_fake_judge() -> None:
    calls: list[str] = []

    def fake_judge(case, result) -> JudgeResult:
        calls.append(case.question)
        assert result.answer == "The invoice identifier is ZXQ-7781."
        return JudgeResult(score=0.9, passed=True, reason="Faithful.")

    summary = run_evals(
        pipeline=build_pipeline(),
        tenant_id="tenant-a",
        cases=[EvalCase(question="What is the invoice identifier?")],
        top_k=1,
        judge=fake_judge,
    )

    assert calls == ["What is the invoice identifier?"]
    assert summary.judge_enabled is True
    assert summary.results[0].judge is not None


def test_load_cases_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "What is the invoice identifier?",
                "expected_answer_contains": "ZXQ-7781",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases == [
        EvalCase(
            question="What is the invoice identifier?",
            expected_answer_contains=["ZXQ-7781"],
        )
    ]


def test_eval_runner_accepts_eval_judge_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m evals.run", "resume_remote.pdf", "--eval-judge"],
    )

    args = parse_args()

    assert args.pdf == Path("resume_remote.pdf")
    assert args.eval_judge is True
