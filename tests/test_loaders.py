import io

import pytest

from rag.ingestion.loaders import (
    DocumentParseError,
    UnsupportedFileType,
    load_document,
    split_markdown_sections,
)
from tests.conftest import DOCS


def test_markdown_sections_have_heading_paths():
    md = "# Guide\nintro\n\n## Setup\nstep one\n\n### Linux\napt\n\n## Usage\nrun it\n"
    sections = split_markdown_sections(md)
    assert [s.heading for s in sections] == [
        "Guide",
        "Guide > Setup",
        "Guide > Setup > Linux",
        "Guide > Usage",
    ]
    assert sections[2].text == "apt"


def test_markdown_ignores_headings_inside_code_fences():
    md = "# Title\n```bash\n# not a heading\necho hi\n```\n"
    sections = split_markdown_sections(md)
    assert len(sections) == 1
    assert "# not a heading" in sections[0].text


def test_html_loader_strips_scripts_and_navigation_and_keeps_headings():
    html = b"""<html><head><title>T</title><script>evil()</script></head><body>
    <nav>Menu</nav><h1>Doc</h1><p>Hello <b>bold</b>
    world.</p><h2>Part</h2><ul><li>one</li><li>two</li></ul><footer>foot</footer></body></html>"""
    doc = load_document("page.html", html)
    assert doc.title == "T"
    text = " ".join(s.text for s in doc.sections)
    assert "evil" not in text and "Menu" not in text and "foot" not in text
    assert "Hello bold world." in text
    assert doc.sections[-1].heading == "Doc > Part"
    assert "- one" in doc.sections[-1].text


def test_docx_loader_maps_heading_styles():
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Manual", level=0)
    d.add_heading("Install", level=1)
    d.add_paragraph("Run the installer.")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "key"
    table.rows[0].cells[1].text = "value"
    buf = io.BytesIO()
    d.save(buf)
    doc = load_document("manual.docx", buf.getvalue())
    assert doc.sections[0].heading == "Manual > Install"
    assert "Run the installer." in doc.sections[0].text
    assert "key | value" in doc.sections[-1].text


def test_pdf_loader_reports_pages():
    doc = load_document("security-policy.pdf", (DOCS / "security-policy.pdf").read_bytes())
    assert [s.page for s in doc.sections] == [1, 2]
    assert "14 characters" in doc.sections[0].text
    assert "400 days" in doc.sections[1].text
    assert doc.metadata["pages"] == 2


def test_sample_docs_all_parse():
    for path in DOCS.iterdir():
        doc = load_document(path.name, path.read_bytes())
        assert doc.sections, path.name


def test_unsupported_and_corrupt_files():
    with pytest.raises(UnsupportedFileType):
        load_document("image.png", b"\x89PNG")
    with pytest.raises(DocumentParseError):
        load_document("broken.pdf", b"not a pdf at all")
    with pytest.raises(DocumentParseError):
        load_document("broken.docx", b"not a zip")
