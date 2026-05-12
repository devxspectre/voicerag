from core.chunker import HierarchicalChunker, PageText


def test_chunker_builds_section_parents_and_child_indexes() -> None:
    chunker = HierarchicalChunker(
        parent_token_limit=18,
        child_token_limit=8,
        child_overlap_tokens=2,
    )
    result = chunker.chunk_pages(
        pages=[
            PageText(
                page=1,
                text=(
                    "1.1 Nitrogen Recommendations\n"
                    "Apply nitrogen before transplanting for rice growth.\n"
                    "Use soil test values for final rates."
                ),
            ),
            PageText(
                page=2,
                text=(
                    "1.2 Phosphorus Recommendations\n"
                    "Apply phosphorus during land preparation.\n"
                    "Avoid over application near waterways."
                ),
            ),
        ],
        tenant_id="tenant-a",
        source_file="docs/rice.pdf",
        document_id="doc-rice",
    )

    assert result.parents
    assert result.children
    assert {parent.position.section for parent in result.parents} == {
        "1.1 Nitrogen Recommendations",
        "1.2 Phosphorus Recommendations",
    }
    assert all(child.parent_id for child in result.children)
    assert all(child.source_file == "docs/rice.pdf" for child in result.children)
    assert all(child.tenant_id == "tenant-a" for child in result.children)
    assert [child.position.chunk_index for child in result.children] == list(
        range(len(result.children))
    )
    assert all("section_title" in child.metadata for child in result.children)


def test_chunker_preserves_page_ranges_for_cross_page_section() -> None:
    chunker = HierarchicalChunker(
        parent_token_limit=100,
        child_token_limit=20,
        child_overlap_tokens=3,
    )
    result = chunker.chunk_pages(
        pages=[
            PageText(
                page=3,
                text="2.1 Shared Section\nThis section starts on page three.",
            ),
            PageText(
                page=4,
                text="This same section continues on page four without a new heading.",
            ),
        ],
        tenant_id="tenant-a",
        source_file="docs/rice.pdf",
        document_id="doc-rice",
    )

    assert len(result.parents) == 1
    parent = result.parents[0]
    assert parent.position.page is None
    assert parent.position.page_start == 3
    assert parent.position.page_end == 4
    assert result.children[0].position.page_start == 3
    assert result.children[0].position.page_end == 4


def test_chunker_uses_stable_ids() -> None:
    chunker = HierarchicalChunker()
    pages = [PageText(page=1, text="1.1 Stable Section\nStable text.")]

    first = chunker.chunk_pages(
        pages=pages,
        tenant_id="tenant-a",
        source_file="docs/rice.pdf",
        document_id="doc-rice",
    )
    second = chunker.chunk_pages(
        pages=pages,
        tenant_id="tenant-a",
        source_file="docs/rice.pdf",
        document_id="doc-rice",
    )

    assert [chunk.chunk_id for chunk in first.parents] == [
        chunk.chunk_id for chunk in second.parents
    ]
    assert [chunk.chunk_id for chunk in first.children] == [
        chunk.chunk_id for chunk in second.children
    ]
