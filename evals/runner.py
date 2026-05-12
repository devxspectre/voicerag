from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.pipeline import PipelineQueryResult, RagPipeline


DEFAULT_EVAL_CASES_PATH = Path("evals/cases.jsonl")
DEFAULT_EVAL_RESULTS_PATH = Path("outputs/evals/results.jsonl")
DEFAULT_EVAL_SUMMARY_PATH = Path("outputs/evals/summary.json")
DEFAULT_LIVE_EVAL_RESULTS_PATH = Path("outputs/evals/live_results.jsonl")


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_answer_contains: list[str] = field(default_factory=list)
    expected_child_ids: list[str] = field(default_factory=list)
    expected_parent_ids: list[str] = field(default_factory=list)
    expected_source_file: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCase":
        return cls(
            question=str(payload["question"]),
            expected_answer_contains=_string_list(payload.get("expected_answer_contains")),
            expected_child_ids=_string_list(payload.get("expected_child_ids")),
            expected_parent_ids=_string_list(payload.get("expected_parent_ids")),
            expected_source_file=payload.get("expected_source_file"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True)
class JudgeResult:
    score: float
    passed: bool
    reason: str


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float | None
    recall_at_k: float | None
    mrr: float | None
    hit_rate: float | None
    relevant_total: int
    retrieved_total: int
    first_relevant_rank: int | None


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    passed: bool
    checks: dict[str, bool]
    answer: str | None
    child_ids: list[str]
    parent_ids: list[str]
    source_files: list[str]
    judge: JudgeResult | None = None
    metrics: RetrievalMetrics | None = None


@dataclass(frozen=True)
class EvalSummary:
    total: int
    passed: int
    failed: int
    judge_enabled: bool
    results: list[EvalResult]


Judge = Callable[[EvalCase, PipelineQueryResult], JudgeResult]


def load_cases(path: Path = DEFAULT_EVAL_CASES_PATH) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(
            f"Eval cases file not found: {path}. Create one JSONL case per line, "
            'for example: {"question":"What is this document about?",'
            '"expected_answer_contains":["keyword"]}'
        )

    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            cases.append(EvalCase.from_dict(json.loads(stripped)))
    return cases


def write_eval_outputs(
    summary: EvalSummary,
    results_path: Path = DEFAULT_EVAL_RESULTS_PATH,
    summary_path: Path = DEFAULT_EVAL_SUMMARY_PATH,
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as file:
        for result in summary.results:
            file.write(json.dumps(_to_payload(result), sort_keys=True) + "\n")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "judge_enabled": summary.judge_enabled,
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def judge_result_from_model_output(raw: str) -> JudgeResult:
    try:
        payload = json.loads(_extract_json(raw))
        score = float(payload.get("score", 0.0))
        passed = bool(payload.get("passed", score >= 0.7))
        reason = str(payload.get("reason", "No reason provided."))
    except Exception:
        score = 0.0
        passed = False
        reason = f"Judge returned unparsable output: {raw}"
    return JudgeResult(score=score, passed=passed, reason=reason)


def build_judge_prompt(question: str, result: PipelineQueryResult) -> str:
    return (
        "Grade this RAG answer as JSON only with keys score, passed, reason. "
        "Score is 0 to 1. Passed should be true only when the answer is "
        "faithful to the retrieved context and relevant to the question.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{result.context}\n\n"
        f"Answer:\n{result.answer or ''}"
    )


def write_live_eval_result(
    question: str,
    result: PipelineQueryResult,
    judge: JudgeResult | None,
    metrics: RetrievalMetrics | None = None,
    matched_case: EvalCase | None = None,
    output_path: Path = DEFAULT_LIVE_EVAL_RESULTS_PATH,
    error: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_unix": time.time(),
        "question": question,
        "answer": result.answer,
        "context_length": len(result.context),
        "citations": [
            {
                "child_id": citation.child_id,
                "parent_id": citation.parent_id,
                "source_file": citation.source_file,
                "position": citation.position,
                "score": citation.score,
                "dense_score": citation.dense_score,
                "bm25_score": citation.bm25_score,
            }
            for citation in result.citations
        ],
        "judge": _to_payload(judge) if judge else None,
        "metrics": _to_payload(metrics) if metrics else None,
        "matched_case": _to_payload(matched_case) if matched_case else None,
        "error": error,
    }
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_to_payload(payload), sort_keys=True) + "\n")


def run_evals(
    pipeline: RagPipeline,
    tenant_id: str,
    cases: Iterable[EvalCase],
    top_k: int,
    judge: Judge | None = None,
) -> EvalSummary:
    results: list[EvalResult] = []
    for case in cases:
        query_result = pipeline.query(
            tenant_id=tenant_id,
            question=case.question,
            top_k=top_k,
        )
        checks = _deterministic_checks(case, query_result)
        metrics = compute_retrieval_metrics(case, query_result)
        judge_result = judge(case, query_result) if judge else None
        passed = all(checks.values()) and (judge_result.passed if judge_result else True)
        results.append(
            EvalResult(
                case=case,
                passed=passed,
                checks=checks,
                answer=query_result.answer,
                child_ids=[citation.child_id for citation in query_result.citations],
                parent_ids=[citation.parent_id for citation in query_result.citations],
                source_files=[citation.source_file for citation in query_result.citations],
                judge=judge_result,
                metrics=metrics,
            )
        )

    passed_count = sum(1 for result in results if result.passed)
    return EvalSummary(
        total=len(results),
        passed=passed_count,
        failed=len(results) - passed_count,
        judge_enabled=judge is not None,
        results=results,
    )


def _deterministic_checks(
    case: EvalCase,
    result: PipelineQueryResult,
) -> dict[str, bool]:
    answer = result.answer or ""
    child_ids = {citation.child_id for citation in result.citations}
    parent_ids = {citation.parent_id for citation in result.citations}
    source_files = {citation.source_file for citation in result.citations}
    return {
        "answer_contains": all(
            expected.lower() in answer.lower()
            for expected in case.expected_answer_contains
        ),
        "child_ids": set(case.expected_child_ids).issubset(child_ids),
        "parent_ids": set(case.expected_parent_ids).issubset(parent_ids),
        "source_file": (
            True
            if case.expected_source_file is None
            else _source_file_matches(case.expected_source_file, source_files)
        ),
    }


def compute_retrieval_metrics(
    case: EvalCase | None,
    result: PipelineQueryResult,
) -> RetrievalMetrics | None:
    if case is None:
        return None

    relevant_ids = set(case.expected_child_ids) | set(case.expected_parent_ids)
    if not relevant_ids:
        return None

    retrieved_ids: list[str] = []
    for citation in result.citations:
        retrieved_ids.append(citation.child_id)
        retrieved_ids.append(citation.parent_id)

    if not retrieved_ids:
        return RetrievalMetrics(
            precision_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            hit_rate=0.0,
            relevant_total=len(relevant_ids),
            retrieved_total=0,
            first_relevant_rank=None,
        )

    first_relevant_rank = None
    relevant_retrieved: set[str] = set()
    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id not in relevant_ids:
            continue
        relevant_retrieved.add(retrieved_id)
        if first_relevant_rank is None:
            first_relevant_rank = rank

    precision = len(relevant_retrieved) / len(retrieved_ids)
    recall = len(relevant_retrieved) / len(relevant_ids)
    return RetrievalMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        mrr=0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        hit_rate=1.0 if first_relevant_rank is not None else 0.0,
        relevant_total=len(relevant_ids),
        retrieved_total=len(retrieved_ids),
        first_relevant_rank=first_relevant_rank,
    )


def find_matching_case(question: str, cases: Iterable[EvalCase]) -> EvalCase | None:
    normalized = _normalize_question(question)
    for case in cases:
        if _normalize_question(case.question) == normalized:
            return case
    return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _source_file_matches(expected: str, actual_files: set[str]) -> bool:
    expected_path = Path(expected)
    for actual in actual_files:
        actual_path = Path(actual)
        if actual == expected:
            return True
        if actual_path.name == expected_path.name:
            return True
    return False


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _to_payload(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _to_payload(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_payload(item) for item in value]
    return value
