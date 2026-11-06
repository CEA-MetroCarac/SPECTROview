"""
Unit tests for spectroview.view.components.markdown_utils.
"""

import markdown
from spectroview.view.components.markdown_utils import ensure_markdown_spacing


def test_ensure_markdown_spacing_empty():
    assert ensure_markdown_spacing("") == ""
    assert ensure_markdown_spacing(None) is None


def test_ensure_markdown_spacing_unordered_list():
    raw = (
        "Here are the options:\n"
        "- Option A\n"
        "- Option B"
    )
    fixed = ensure_markdown_spacing(raw)
    assert fixed == (
        "Here are the options:\n\n"
        "- Option A\n"
        "- Option B"
    )
    # Verify Python-Markdown parses it as a list with <ul> and <li>
    html = markdown.markdown(fixed)
    assert "<ul>" in html
    assert "<li>Option A</li>" in html
    assert "<li>Option B</li>" in html


def test_ensure_markdown_spacing_ordered_list():
    raw = (
        "Steps to perform:\n"
        "1. First step\n"
        "2. Second step"
    )
    fixed = ensure_markdown_spacing(raw)
    assert fixed == (
        "Steps to perform:\n\n"
        "1. First step\n"
        "2. Second step"
    )
    html = markdown.markdown(fixed)
    assert "<ol>" in html
    assert "<li>First step</li>" in html


def test_ensure_markdown_spacing_already_spaced():
    raw = (
        "Intro text:\n\n"
        "- Item 1\n"
        "- Item 2"
    )
    fixed = ensure_markdown_spacing(raw)
    assert fixed == raw


def test_ensure_markdown_spacing_heading_before_list():
    raw = (
        "### Section Header\n"
        "- Item 1\n"
        "- Item 2"
    )
    fixed = ensure_markdown_spacing(raw)
    # Headers followed directly by lists do not need extra blank line
    assert fixed == raw


def test_ensure_markdown_spacing_code_fence():
    raw = (
        "```python\n"
        "# Code sample\n"
        "- not a list item\n"
        "1. also not a list\n"
        "```"
    )
    fixed = ensure_markdown_spacing(raw)
    assert fixed == raw


def test_ensure_markdown_spacing_nested_list():
    raw = (
        "Intro:\n"
        "- Parent 1\n"
        "  - Child 1\n"
        "  - Child 2\n"
        "- Parent 2"
    )
    fixed = ensure_markdown_spacing(raw)
    assert fixed == (
        "Intro:\n\n"
        "- Parent 1\n"
        "  - Child 1\n"
        "  - Child 2\n"
        "- Parent 2"
    )


def test_ensure_markdown_spacing_table():
    raw = (
        "Table title:\n"
        "| Col 1 | Col 2 |\n"
        "|-------|-------|\n"
        "| Val 1 | Val 2 |"
    )
    fixed = ensure_markdown_spacing(raw)
    assert fixed == (
        "Table title:\n\n"
        "| Col 1 | Col 2 |\n"
        "|-------|-------|\n"
        "| Val 1 | Val 2 |"
    )
    html = markdown.markdown(fixed, extensions=['tables'])
    assert "<table>" in html


def test_ensure_markdown_spacing_blockquote():
    raw = (
        "> Quote intro:\n"
        "> - Bullet A\n"
        "> - Bullet B"
    )
    fixed = ensure_markdown_spacing(raw)
    assert fixed == (
        "> Quote intro:\n"
        ">\n"
        "> - Bullet A\n"
        "> - Bullet B"
    )
    html = markdown.markdown(fixed)
    assert "<blockquote>" in html
    assert "<ul>" in html


def test_section_3_5_spectra_maps_regression():
    raw = (
        "In both the **Spectra** and **Maps** workspaces, you can interactively bracket up to three spectral features in the `SpectraViewer` and inspect their variation in real time within the dedicated **Live Maps** tab:\n"
        "- **Spectra Workspace**: Displays up to three color-coded (Red, Blue, Green) **1D parameter variation graphs**\n"
        "- **Maps Workspace**: Displays up to three real-time **2D spatial heatmaps**"
    )
    fixed = ensure_markdown_spacing(raw)
    html = markdown.markdown(fixed)
    assert "<ul>" in html
    assert "<li>" in html
    assert "Spectra Workspace" in html
    assert "Maps Workspace" in html

