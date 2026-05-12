from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


NUMBERED_HEADING_PATTERN = re.compile(
    r"^\s*((chapter|section)\s+\d+(\.\d+)*|\d+(\.\d+)+\.?)\s+.+$",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"\S+")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkPosition:
    """Traceable location for a chunk inside its source document."""

    chunk_index: int
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    start_token: int | None = None
    end_token: int | None = None
    section: str | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"chunk_index": self.chunk_index}
        if self.page is not None:
            payload["page"] = self.page
        if self.page_start is not None:
            payload["page_start"] = self.page_start
        if self.page_end is not None:
            payload["page_end"] = self.page_end
        if self.start_token is not None:
            payload["start_token"] = self.start_token
        if self.end_token is not None:
            payload["end_token"] = self.end_token
        if self.section is not None:
            payload["section"] = self.section
        return payload


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    tenant_id: str
    source_file: str
    document_id: str
    position: ChunkPosition
    parent_id: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_parent(self) -> bool:
        return self.parent_id is None

    @property
    def is_child(self) -> bool:
        return self.parent_id is not None

    def require_valid(self) -> None:
        missing = [
            name
            for name, value in (
                ("chunk_id", self.chunk_id),
                ("tenant_id", self.tenant_id),
                ("source_file", self.source_file),
                ("document_id", self.document_id),
                ("text", self.text),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Chunk missing required fields: {', '.join(missing)}")


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class ChunkingResult:
    parents: list[Chunk]
    children: list[Chunk]


@dataclass
class _SectionBuffer:
    title: str
    page_start: int
    page_end: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line for line in self.lines if line.strip()).strip()


@dataclass(frozen=True)
class _TokenWindow:
    text: str
    start_token: int
    end_token: int


class HierarchicalChunker:
    def __init__(
        self,
        parent_token_limit: int = 1600,
        child_token_limit: int = 500,
        child_overlap_tokens: int = 75,
        min_heading_chars: int = 4,
    ) -> None:
        if parent_token_limit < 1:
            raise ValueError("parent_token_limit must be at least 1")
        if child_token_limit < 1:
            raise ValueError("child_token_limit must be at least 1")
        if child_overlap_tokens >= child_token_limit:
            raise ValueError("child_overlap_tokens must be smaller than child_token_limit")

        self.parent_token_limit = parent_token_limit
        self.child_token_limit = child_token_limit
        self.child_overlap_tokens = child_overlap_tokens
        self.min_heading_chars = min_heading_chars

    def chunk_pages(
        self,
        pages: list[PageText],
        tenant_id: str,
        source_file: str,
        document_id: str | None = None,
    ) -> ChunkingResult:
        document_id = document_id or stable_document_id(source_file)
        sections = self._sections_from_pages(pages)

        parent_chunks: list[Chunk] = []
        child_chunks: list[Chunk] = []
        parent_index = 0
        child_index = 0

        for section in sections:
            for parent_window_index, parent_window in enumerate(
                split_token_windows(section.text, self.parent_token_limit, overlap_tokens=0)
            ):
                parent_id = stable_chunk_id(
                    tenant_id,
                    document_id,
                    "parent",
                    parent_index,
                    section.title,
                    parent_window.text,
                )
                parent_position = ChunkPosition(
                    chunk_index=parent_index,
                    page=_single_page_or_none(section.page_start, section.page_end),
                    page_start=section.page_start,
                    page_end=section.page_end,
                    start_token=parent_window.start_token,
                    end_token=parent_window.end_token,
                    section=section.title,
                )
                parent = Chunk(
                    chunk_id=parent_id,
                    tenant_id=tenant_id,
                    source_file=source_file,
                    document_id=document_id,
                    position=parent_position,
                    parent_id=None,
                    text=parent_window.text,
                    metadata={
                        "level": "parent",
                        "section_title": section.title,
                        "parent_window_index": parent_window_index,
                    },
                )
                parent_chunks.append(parent)

                for child_window in split_token_windows(
                    parent_window.text,
                    self.child_token_limit,
                    overlap_tokens=self.child_overlap_tokens,
                ):
                    child_id = stable_chunk_id(
                        tenant_id,
                        document_id,
                        "child",
                        child_index,
                        parent_id,
                        child_window.text,
                    )
                    child_chunks.append(
                        Chunk(
                            chunk_id=child_id,
                            tenant_id=tenant_id,
                            source_file=source_file,
                            document_id=document_id,
                            position=ChunkPosition(
                                chunk_index=child_index,
                                page=parent_position.page,
                                page_start=section.page_start,
                                page_end=section.page_end,
                                start_token=child_window.start_token,
                                end_token=child_window.end_token,
                                section=section.title,
                            ),
                            parent_id=parent_id,
                            text=child_window.text,
                            metadata={
                                "level": "child",
                                "section_title": section.title,
                                "parent_chunk_index": parent_index,
                            },
                        )
                    )
                    child_index += 1

                parent_index += 1

        return ChunkingResult(parents=parent_chunks, children=child_chunks)

    def _sections_from_pages(self, pages: list[PageText]) -> list[_SectionBuffer]:
        sections: list[_SectionBuffer] = []
        current: _SectionBuffer | None = None

        for page in pages:
            for raw_line in page.text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                if self._is_heading(line):
                    if current and current.text:
                        sections.append(current)
                    current = _SectionBuffer(
                        title=line,
                        page_start=page.page,
                        page_end=page.page,
                        lines=[line],
                    )
                    continue

                if current is None:
                    current = _SectionBuffer(
                        title="Document Start",
                        page_start=page.page,
                        page_end=page.page,
                    )
                current.page_end = page.page
                current.lines.append(line)

        if current and current.text:
            sections.append(current)

        return sections

    def _is_heading(self, line: str) -> bool:
        if len(line) < self.min_heading_chars:
            return False
        if len(line.split()) > 12:
            return False
        if NUMBERED_HEADING_PATTERN.match(line):
            return True
        letters = [char for char in line if char.isalpha()]
        return bool(letters) and sum(char.isupper() for char in letters) / len(letters) > 0.8


def split_token_windows(
    text: str,
    token_limit: int,
    overlap_tokens: int,
) -> list[_TokenWindow]:
    tokens = TOKEN_PATTERN.findall(text)
    if not tokens:
        return []
    if len(tokens) <= token_limit:
        return [_TokenWindow(text=text.strip(), start_token=0, end_token=len(tokens))]

    windows: list[_TokenWindow] = []
    start = 0
    while start < len(tokens):
        end = min(start + token_limit, len(tokens))
        window_tokens = tokens[start:end]
        windows.append(
            _TokenWindow(
                text=" ".join(window_tokens),
                start_token=start,
                end_token=end,
            )
        )
        if end == len(tokens):
            break
        start = max(end - overlap_tokens, start + 1)
    return windows


def stable_document_id(source_file: str) -> str:
    digest = hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:16]
    return f"doc-{digest}"


def stable_chunk_id(
    tenant_id: str,
    document_id: str,
    level: str,
    index: int,
    label: str,
    text: str,
) -> str:
    digest = hashlib.sha256(
        "\n".join([tenant_id, document_id, level, str(index), label, text]).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return f"{document_id}-{level}-{index}-{digest}"


def _single_page_or_none(page_start: int, page_end: int) -> int | None:
    if page_start == page_end:
        return page_start
    return None
