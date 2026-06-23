"""Extraction layer: convert files, URLs, or HTML strings into Markdown."""

from pathlib import Path
from datetime import datetime, timezone
from markitdown import MarkItDown
import httpx
import re


_md_converter = MarkItDown()


def extract_from_file(
    file_path: str | Path,
    converter: MarkItDown | None = None,
    strip_tables: bool = True,
) -> dict | None:
    """Convert a local file to Markdown and return it with source metadata."""
    conv = converter or _md_converter
    path = Path(file_path)
    if not path.is_file():
        print(f"  [SKIP] not a file: {path}")
        return None

    try:
        result = conv.convert(str(path))
        md_text = getattr(result, "markdown", "")
        title = getattr(result, "title", None) or path.stem
    except Exception as exc:
        print(f"  [FAIL] markitdown error for {path.name}: {exc}")
        return None

    if not md_text or not md_text.strip():
        print(f"  [SKIP] empty output: {path.name}")
        return None

    if strip_tables:
        md_text = strip_markdown_tables(md_text)

    metadata = {
        "source": str(path.resolve()),
        "title": title,
        "doc_type": path.suffix.lower(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f"  [OK]   {path.name}  ({len(md_text)} chars)")
    return {"markdown": md_text, "metadata": metadata}


def extract_from_url(url: str) -> dict | None:
    """Fetch a web page and pass the HTML through the same extraction path."""
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        html = resp.text
    except Exception as exc:
        print(f"  [FAIL] http error for {url}: {exc}")
        return None

    return extract_from_html(html, source_url=url)


def extract_from_html(
    html: str,
    source_url: str = "inline.html",
    converter: MarkItDown | None = None,
    strip_tables: bool = True,
) -> dict | None:
    """Convert an HTML string to Markdown and attach URL-like metadata."""
    conv = converter or _md_converter
    try:
        result = conv.convert(html, file_extension=".html")
        md_text = getattr(result, "markdown", "")
        title = getattr(result, "title", None) or Path(source_url).stem
    except Exception as exc:
        print(f"  [FAIL] markitdown html error: {exc}")
        return None

    if not md_text or not md_text.strip():
        print(f"  [SKIP] empty html output: {source_url}")
        return None

    if strip_tables:
        md_text = strip_markdown_tables(md_text)

    metadata = {
        "source": source_url,
        "title": title,
        "doc_type": ".html",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f"  [OK]   {source_url}  ({len(md_text)} chars)")
    return {"markdown": md_text, "metadata": metadata}


def strip_markdown_tables(markdown: str) -> str:
    """Remove markdown table blocks that often pollute PDF-derived chunks.

    MarkItDown can produce large pipe-delimited tables from PDFs. Those rows
    usually retrieve poorly, so this keeps normal prose while dropping complete
    markdown table blocks.
    """
    lines = markdown.splitlines()
    cleaned = []
    pending = None
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_like = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2
        is_separator = bool(re.match(r"^\|[\s\-:|]+\|$", stripped))

        # A table is confirmed only when a pipe row is followed by a separator.
        if pending is not None and is_separator:
            pending = None
            in_table = True
            continue

        # Once inside a confirmed table, skip table rows and blank spacer lines.
        if in_table:
            if is_table_like or not stripped:
                continue
            in_table = False

        # If the previous pipe-like line was not a table header, keep it.
        if pending is not None:
            cleaned.append(pending)
            pending = None

        # Hold pipe-like lines until the next line tells us whether it is a table.
        if is_table_like:
            pending = line
            continue
        cleaned.append(line)

    if pending is not None:
        cleaned.append(pending)

    return "\n".join(cleaned).strip()
