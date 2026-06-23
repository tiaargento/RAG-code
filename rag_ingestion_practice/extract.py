from pathlib import Path
from datetime import datetime, timezone
from markitdown import MarkItDown
import httpx


_md_converter = MarkItDown()


def extract_from_file(file_path: str | Path, converter: MarkItDown | None = None) -> dict | None:
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

    metadata = {
        "source": str(path.resolve()),
        "title": title,
        "doc_type": path.suffix.lower(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f"  [OK]   {path.name}  ({len(md_text)} chars)")
    return {"markdown": md_text, "metadata": metadata}


def extract_from_url(url: str) -> dict | None:
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        html = resp.text
    except Exception as exc:
        print(f"  [FAIL] http error for {url}: {exc}")
        return None

    return extract_from_html(html, source_url=url)


def extract_from_html(html: str, source_url: str = "inline.html", converter: MarkItDown | None = None) -> dict | None:
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

    metadata = {
        "source": source_url,
        "title": title,
        "doc_type": ".html",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f"  [OK]   {source_url}  ({len(md_text)} chars)")
    return {"markdown": md_text, "metadata": metadata}
