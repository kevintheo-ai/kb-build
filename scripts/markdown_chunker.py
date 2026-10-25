"""
markdown_chunker.py - split a Markdown file into coherent chunks.

Heading-aware first (H1-H4 become metadata), then a recursive character split for sections
that are still too long. Front matter is not part of any chunk.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS_TO_SPLIT = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


def _content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_frontmatter(text: str) -> str:
    text = text.lstrip("\ufeff")  # a UTF-8 BOM would hide the opening --- (same as parse_frontmatter)
    if text.startswith("---"):
        match = re.match(r"^---\s*\n.*?\n---\s*\n?", text, re.DOTALL)
        if match:
            return text[match.end() :]
    return text


def chunk_markdown_file(file_path: Path, max_chunk_size: int = 1000, chunk_overlap: int = 120) -> list[dict]:
    """-> [{"text": ..., "metadata": {file_path, h1, h2, h3, h4, chunk_index, content_hash}}]"""
    path = Path(file_path)
    content = _strip_frontmatter(path.read_text(encoding="utf-8", errors="replace")).strip()
    if not content:
        return []

    header_chunks = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT, strip_headers=False).split_text(
        content
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    if header_chunks:
        docs = splitter.split_documents(header_chunks)
    else:
        from langchain_core.documents import Document

        docs = splitter.split_documents([Document(page_content=content, metadata={})])

    result: list[dict] = []
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
        result.append(
            {
                "text": text,
                "metadata": {
                    "file_path": str(path.resolve()),
                    "h1": doc.metadata.get("h1", ""),
                    "h2": doc.metadata.get("h2", ""),
                    "h3": doc.metadata.get("h3", ""),
                    "h4": doc.metadata.get("h4", ""),
                    "chunk_index": len(result),
                    "content_hash": _content_hash(text),
                },
            }
        )
    return result
