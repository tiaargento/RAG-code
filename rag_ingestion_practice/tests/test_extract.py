from pathlib import Path
from extract import extract_from_file, extract_from_html, strip_markdown_tables


class TestExtractFromFile:
    def test_extract_txt_file(self, tmp_dir):
        """A real .txt file returns markdown and metadata."""
        f = tmp_dir / "hello.txt"
        f.write_text("Hello world", encoding="utf-8")

        result = extract_from_file(str(f))
        assert result is not None
        assert "markdown" in result
        assert "metadata" in result
        assert "Hello world" in result["markdown"]
        assert result["metadata"]["title"] == "hello"
        assert result["metadata"]["doc_type"] == ".txt"
        assert "timestamp" in result["metadata"]

    def test_extract_nonexistent_file(self):
        """Non-existent file returns None."""
        result = extract_from_file("/does/not/exist.txt")
        assert result is None

    def test_extract_empty_file(self, tmp_dir):
        """Empty file returns None."""
        f = tmp_dir / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = extract_from_file(str(f))
        assert result is None

    def test_extract_with_mock_converter(self, tmp_dir, mock_converter):
        """Injected mock converter is used."""
        f = tmp_dir / "test.txt"
        f.write_text("dummy", encoding="utf-8")
        result = extract_from_file(str(f), converter=mock_converter)
        assert result is not None
        assert "Mocked Title" in result["markdown"]
        assert result["metadata"]["title"] == "Mocked Title"


class TestExtractFromHtml:
    def test_extract_simple_html(self, mock_converter):
        """Inline HTML returns markdown via converter."""
        html = "<html><body><h1>Hello</h1></body></html>"
        result = extract_from_html(html, source_url="test.html", converter=mock_converter)
        assert result is not None
        assert "markdown" in result
        assert "metadata" in result
        assert result["metadata"]["source"] == "test.html"
        assert result["metadata"]["doc_type"] == ".html"

    def test_extract_html_empty(self, tmp_dir, empty_mock_converter):
        """Empty HTML returns None when converter yields empty output."""
        result = extract_from_html("", source_url="empty.html", converter=empty_mock_converter)
        assert result is None


class TestMarkdownCleanup:
    def test_strip_markdown_tables_removes_table_block(self):
        markdown = (
            "Before table\n\n"
            "| Col A | Col B |\n"
            "| --- | --- |\n"
            "| one | two |\n"
            "| three | four |\n\n"
            "After table"
        )
        cleaned = strip_markdown_tables(markdown)
        assert "Before table" in cleaned
        assert "After table" in cleaned
        assert "| Col A | Col B |" not in cleaned
        assert "| one | two |" not in cleaned

    def test_strip_markdown_tables_keeps_non_table_pipe_line(self):
        markdown = "This line mentions A | B but is not a markdown table."
        assert strip_markdown_tables(markdown) == markdown
