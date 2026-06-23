"""Chunking layer: split Markdown into retrieval-sized pieces."""

import re
import hashlib
from datetime import datetime, timezone
import tiktoken

from config import CHUNK_SIZE, CHUNK_OVERLAP


_enc = tiktoken.get_encoding("cl100k_base")


def _num_tokens(text: str) -> int:
    """Count tokens using the same encoding used by the chunk splitter."""
    return len(_enc.encode(text, disallowed_special=()))


def chunk_markdown(markdown: str, metadata: dict) -> list[dict]:
    """Split Markdown into chunks and attach retrieval metadata to each chunk."""
    sections = _split_by_headings(markdown)

    chunks = []
    for heading, body in sections:
        section_text = f"{heading}\n{body}" if heading else body
        tokens = _num_tokens(section_text)

        # Keep short heading sections intact so headings stay with their content.
        if tokens <= CHUNK_SIZE:
            chunks.append(section_text)
        else:
            # For long sections, prefer paragraph boundaries before token cuts.
            paragraphs = re.split(r"\n\s*\n", section_text)
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                candidate = current + "\n\n" + para if current else para
                if _num_tokens(candidate) <= CHUNK_SIZE:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    if _num_tokens(para) > CHUNK_SIZE:
                        # Very long paragraphs still need a token-level fallback.
                        chunks.extend(_split_by_tokens(para))
                    else:
                        current = para
            if current:
                chunks.append(current)

    chunk_docs = []
    for idx, text in enumerate(chunks):
        # The chunk hash is used by store.py for simple re-ingestion deduping.
        doc = {
            "text": text,
            "metadata": {
                "source": metadata.get("source", ""),
                "title": metadata.get("title", ""),
                "doc_type": metadata.get("doc_type", ""),
                "chunk_index": idx,
                "timestamp": metadata.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "chunk_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            },
        }
        chunk_docs.append(doc)

    return chunk_docs


def _split_by_tokens(text: str) -> list[str]:
    """Split oversized text into token windows with overlap."""
    tokens = _enc.encode(text, disallowed_special=())
    parts = []
    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):
        segment = tokens[i : i + CHUNK_SIZE]
        parts.append(_enc.decode(segment))
    return parts


def _split_by_headings(md: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading, body) sections using Markdown headings."""
    pattern = re.compile(r"^(#{1,6}\s+.*)$", re.MULTILINE)
    parts = pattern.split(md)
    sections = []

    if not parts[0].strip():
        parts = parts[1:]

    if parts and not re.match(r"^#{1,6}\s+", parts[0]):
        sections.append(("", parts[0].strip()))

    for i, part in enumerate(parts):
        if re.match(r"^#{1,6}\s+", part):
            if sections:
                sections[-1] = (sections[-1][0], sections[-1][1])
            heading = part.strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections.append((heading, body))

    return sections
