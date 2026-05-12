from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from core import HierarchicalChunker, RagPipeline
from core.pipeline import PipelineQueryResult
from evals.runner import (
    EvalCase,
    JudgeResult,
    build_judge_prompt,
    compute_retrieval_metrics,
    find_matching_case,
    judge_result_from_model_output,
    load_cases,
    write_live_eval_result,
)
from main import build_embedder, extract_pdf_pages, normalize_embedding_model, print_sources
from observability import ObservabilityRecorder
from router.llm_router import DeepSeekRouter


LiveJudge = Callable[[str, PipelineQueryResult], JudgeResult]


def main() -> None:
    load_dotenv()
    args = parse_args()
    pipeline = build_pipeline(args)
    judge = DeepSeekLiveJudge(model_tier=args.model_tier)
    cases = load_cases()
    chat_loop_with_evals(
        pipeline=pipeline,
        tenant_id=args.tenant_id,
        top_k=args.top_k,
        judge=judge,
        cases=cases,
    )


class DeepSeekLiveJudge:
    def __init__(self, model_tier: str = "smart") -> None:
        self.router = DeepSeekRouter(model_tier=model_tier)

    def __call__(self, question: str, result: PipelineQueryResult) -> JudgeResult:
        return judge_result_from_model_output(
            self.router.generate(build_judge_prompt(question, result))
        )


def build_pipeline(args: argparse.Namespace) -> RagPipeline:
    pages = extract_pdf_pages(args.pdf.resolve())
    if not pages:
        raise SystemExit(f"No extractable text found in {args.pdf}")

    answer_generator = DeepSeekRouter(model_tier=args.model_tier).generate
    pipeline = RagPipeline(
        embedder=build_embedder(args.embedder, args.embedding_model),
        answer_generator=answer_generator,
        observability=ObservabilityRecorder(),
    )
    pipeline.index_pages(
        pages=pages,
        tenant_id=args.tenant_id,
        source_file=str(args.pdf.resolve()),
        document_id=args.document_id,
        chunker=HierarchicalChunker(
            parent_token_limit=args.parent_tokens,
            child_token_limit=args.child_tokens,
            child_overlap_tokens=args.child_overlap,
        ),
    )
    return pipeline


def chat_loop_with_evals(
    pipeline: RagPipeline,
    tenant_id: str,
    top_k: int,
    judge: LiveJudge,
    cases: list[EvalCase] | None = None,
) -> None:
    print("Ask questions about the PDF. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue

        answer_once_with_eval(
            pipeline=pipeline,
            tenant_id=tenant_id,
            question=question,
            top_k=top_k,
            judge=judge,
            cases=cases,
        )


def answer_once_with_eval(
    pipeline: RagPipeline,
    tenant_id: str,
    question: str,
    top_k: int,
    judge: LiveJudge,
    cases: list[EvalCase] | None = None,
) -> PipelineQueryResult:
    result = pipeline.query(
        tenant_id=tenant_id,
        question=question,
        top_k=top_k,
    )

    if result.answer:
        print(f"\nMirinda> {result.answer}")
    else:
        print("\nMirinda> LLM generation is disabled. Retrieved context is below.")
    print_sources(result.citations)
    matched_case = find_matching_case(question, cases or [])
    metrics = compute_retrieval_metrics(matched_case, result)

    try:
        verdict = judge(question, result)
    except Exception as exc:
        print(f"\nEval> ERROR - {exc}")
        print(f"Metrics> {_format_metrics(metrics)}")
        write_live_eval_result(
            question=question,
            result=result,
            judge=None,
            metrics=metrics,
            matched_case=matched_case,
            error=str(exc),
        )
        return result

    status = "PASS" if verdict.passed else "FAIL"
    print(f"\nEval> {status} score={verdict.score:.2f} - {verdict.reason}")
    print(f"Metrics> {_format_metrics(metrics)}")
    write_live_eval_result(
        question=question,
        result=result,
        judge=verdict,
        metrics=metrics,
        matched_case=matched_case,
    )
    return result


def _format_metrics(metrics) -> str:
    if metrics is None:
        return "precision@k=n/a recall@k=n/a mrr=n/a hit_rate=n/a"
    return (
        f"precision@k={metrics.precision_at_k:.2f} "
        f"recall@k={metrics.recall_at_k:.2f} "
        f"mrr={metrics.mrr:.2f} "
        f"hit_rate={metrics.hit_rate:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a PDF and judge every answer against retrieved context.",
    )
    parser.add_argument("pdf", type=Path, help="PDF to index.")
    parser.add_argument("--tenant-id", default="local", help="Tenant namespace.")
    parser.add_argument("--document-id", help="Optional stable document ID.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedder",
        choices=["transformers", "mistral", "hash"],
        default="transformers",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--model-tier", choices=["fast", "smart"], default="fast")
    parser.add_argument("--parent-tokens", type=int, default=1600)
    parser.add_argument("--child-tokens", type=int, default=500)
    parser.add_argument("--child-overlap", type=int, default=75)
    args = parser.parse_args()
    args.embedding_model = normalize_embedding_model(args.embedding_model)
    return args


if __name__ == "__main__":
    main()
