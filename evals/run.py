from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from core import HierarchicalChunker, RagPipeline
from evals.runner import (
    EvalCase,
    JudgeResult,
    build_judge_prompt,
    judge_result_from_model_output,
    load_cases,
    run_evals,
    write_eval_outputs,
)
from main import build_embedder, extract_pdf_pages, normalize_embedding_model
from observability import ObservabilityRecorder
from router.llm_router import DeepSeekRouter


def main() -> None:
    load_dotenv()
    args = parse_args()
    pages = extract_pdf_pages(args.pdf.resolve())
    if not pages:
        raise SystemExit(f"No extractable text found in {args.pdf}")

    answer_generator = None
    if not args.no_llm:
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

    judge = DeepSeekJudge(model_tier=args.model_tier) if args.eval_judge else None
    summary = run_evals(
        pipeline=pipeline,
        tenant_id=args.tenant_id,
        cases=load_cases(),
        top_k=args.top_k,
        judge=judge,
    )
    write_eval_outputs(summary)
    print(
        "Eval summary: "
        f"{summary.passed}/{summary.total} passed, "
        f"{summary.failed} failed. Results written to outputs/evals."
    )


class DeepSeekJudge:
    def __init__(self, model_tier: str = "smart") -> None:
        self.router = DeepSeekRouter(model_tier=model_tier)

    def __call__(self, case: EvalCase, result) -> JudgeResult:
        prompt = build_judge_prompt(case.question, result)
        raw = self.router.generate(prompt)
        return judge_result_from_model_output(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predefined RAG evals against a PDF.",
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
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--parent-tokens", type=int, default=1600)
    parser.add_argument("--child-tokens", type=int, default=500)
    parser.add_argument("--child-overlap", type=int, default=75)
    parser.add_argument("--eval-judge", action="store_true")
    args = parser.parse_args()
    args.embedding_model = normalize_embedding_model(args.embedding_model)
    return args


if __name__ == "__main__":
    main()
