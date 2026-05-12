from __future__ import annotations

import argparse
import contextlib
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import pypdf
from pypdf import PdfReader

from core import (
    HashingEmbedder,
    HierarchicalChunker,
    MistralAIEmbedder,
    PageText,
    RagPipeline,
)
from core.embedder import TransformersTextEmbedder
from observability import ObservabilityRecorder
from router.llm_router import DeepSeekRouter


LOGGER = logging.getLogger("rag_cli")


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger(pypdf.__name__).setLevel(logging.ERROR)

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    pages = extract_pdf_pages(pdf_path)
    if not pages:
        raise SystemExit(f"No extractable text found in {pdf_path}")

    answer_generator = None
    if not args.no_llm:
        answer_generator = DeepSeekRouter(model_tier=args.model_tier).generate

    pipeline = RagPipeline(
        embedder=build_embedder(args.embedder, args.embedding_model),
        answer_generator=answer_generator,
        observability=ObservabilityRecorder(),
    )
    chunker = HierarchicalChunker(
        parent_token_limit=args.parent_tokens,
        child_token_limit=args.child_tokens,
        child_overlap_tokens=args.child_overlap,
    )

    LOGGER.info("Indexing %s page(s) from %s", len(pages), pdf_path)
    indexed = pipeline.index_pages(
        pages=pages,
        tenant_id=args.tenant_id,
        source_file=str(pdf_path),
        document_id=args.document_id,
        chunker=chunker,
    )
    LOGGER.info(
        "Indexed %s parent chunk(s) and %s child chunk(s)",
        indexed.parent_count,
        indexed.child_count,
    )

    if args.question:
        answer_once(
            pipeline=pipeline,
            tenant_id=args.tenant_id,
            question=args.question,
            top_k=args.top_k,
            show_context=args.show_context,
        )
        return

    chat_loop(
        pipeline=pipeline,
        tenant_id=args.tenant_id,
        top_k=args.top_k,
        show_context=args.show_context,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a local PDF using the in-memory RAG pipeline.",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=Path("resume_remote.pdf"),
        help="PDF to index. Defaults to sample.pdf.",
    )
    parser.add_argument(
        "-q",
        "--question",
        help="Ask one question and exit. If omitted, starts an interactive chat.",
    )
    parser.add_argument("--tenant-id", default="local", help="Tenant namespace.")
    parser.add_argument("--document-id", help="Optional stable document ID.")
    parser.add_argument("--top-k", type=int, default=5, help="Child matches to retrieve.")
    parser.add_argument(
        "--embedder",
        choices=["transformers", "mistral", "hash"],
        default="transformers",
        help="Embedding backend. Use hash for offline smoke tests.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name for transformers or Mistral backends.",
    )
    parser.add_argument(
        "--model-tier",
        choices=["fast", "smart"],
        default="fast",
        help="DeepSeek model tier for answer generation.",
    )
    parser.add_argument("--parent-tokens", type=int, default=1600)
    parser.add_argument("--child-tokens", type=int, default=500)
    parser.add_argument("--child-overlap", type=int, default=75)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation and print retrieved context/citations.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved parent context with each answer.",
    )
    args = parser.parse_args()
    if args.tenant_id == "local":
        args.tenant_id = os.getenv("DEFAULT_TENANT", args.tenant_id)
    if (
        args.embedder == "mistral"
        and args.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    ):
        args.embedding_model = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
    elif args.embedding_model == "sentence-transformers/all-MiniLM-L6-v2":
        args.embedding_model = os.getenv("EMBEDDING_MODEL", args.embedding_model)
    args.embedding_model = normalize_embedding_model(args.embedding_model)
    return args


def normalize_embedding_model(model_name: str) -> str:
    aliases = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    }
    return aliases.get(model_name, model_name)


def build_embedder(kind: str, model_name: str):
    if kind == "hash":
        LOGGER.info("Using hashing embedder")
        return HashingEmbedder(dimensions=256)
    if kind == "mistral":
        LOGGER.info("Using Mistral AI embedder: %s", model_name)
        return MistralAIEmbedder(model_name=model_name)

    LOGGER.info("Using transformers text embedder: %s", model_name)
    return TransformersTextEmbedder(model_name=model_name)


def extract_pdf_pages(pdf_path: Path) -> list[PageText]:
    pages = extract_pdf_pages_with_pypdf(pdf_path)
    if pages:
        return pages

    LOGGER.warning(
        "pypdf found no usable text in %s; trying PyMuPDF fallback.",
        pdf_path,
    )
    return extract_pdf_pages_with_pymupdf(pdf_path)


def extract_pdf_pages_with_pypdf(pdf_path: Path) -> list[PageText]:
    try:
        reader = PdfReader(str(pdf_path), strict=False)
        total_pages = len(reader.pages)
    except Exception as exc:
        LOGGER.warning("pypdf could not read %s: %s", pdf_path, exc)
        return []

    pages: list[PageText] = []
    for page_index in range(total_pages):
        index = page_index + 1
        try:
            page = reader.pages[page_index]
            text = page.extract_text() or ""
        except Exception as exc:
            LOGGER.warning(
                "Skipping page %s from %s with pypdf: %s",
                index,
                pdf_path,
                exc,
            )
            continue
        cleaned = text.strip()
        if not cleaned:
            LOGGER.warning(
                "Skipping page %s from %s: no extractable text. It may be scanned/OCR-only.",
                index,
                pdf_path,
            )
            continue
        pages.append(PageText(page=index, text=cleaned))
    return pages


def extract_pdf_pages_with_pymupdf(pdf_path: Path) -> list[PageText]:
    import fitz

    pages: list[PageText] = []
    fitz.TOOLS.mupdf_display_errors(False)
    with _quiet_stderr(), fitz.open(pdf_path) as document:
        total_pages = document.page_count
        for page_index in range(total_pages):
            index = page_index + 1
            try:
                page = document[page_index]
                text = page.get_text("text") or ""
            except Exception as exc:
                LOGGER.warning(
                    "Skipping page %s from %s with PyMuPDF: %s",
                    index,
                    pdf_path,
                    exc,
                )
                continue
            cleaned = text.strip()
            if not cleaned:
                LOGGER.warning(
                    "Skipping page %s from %s with PyMuPDF: no extractable text. "
                    "It may need OCR.",
                    index,
                    pdf_path,
                )
                continue
            pages.append(PageText(page=index, text=cleaned))
    return pages


@contextlib.contextmanager
def _quiet_stderr():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stderr(devnull):
            yield


def chat_loop(
    pipeline: RagPipeline,
    tenant_id: str,
    top_k: int,
    show_context: bool,
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

        answer_once(
            pipeline=pipeline,
            tenant_id=tenant_id,
            question=question,
            top_k=top_k,
            show_context=show_context,
        )


def answer_once(
    pipeline: RagPipeline,
    tenant_id: str,
    question: str,
    top_k: int,
    show_context: bool,
) -> None:
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
    if show_context or not result.answer:
        print("\nContext:")
        print(result.context or "No context retrieved.")


def print_sources(citations) -> None:
    if not citations:
        print("\nSources: none")
        return

    print("\nSources:")
    for index, citation in enumerate(citations, start=1):
        position = citation.position
        page = position.get("page") or position.get("page_start")
        section = position.get("section")
        location = f"page {page}" if page else f"position {position}"
        if section:
            location = f"{location}, section {section}"
        print(
            f"{index}. {citation.source_file} ({location}) "
            f"child={citation.child_id} parent={citation.parent_id} "
            f"score={citation.score:.3f}"
        )


if __name__ == "__main__":
    main()
