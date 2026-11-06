"""
Automated markdown syntax validator for SPECTROview documentation.
Ensures that all markdown documents in spectroview/resources/user_manual/ and docs/
maintain proper blank lines before lists and tables to guarantee correct rendering
in Python-Markdown (MkDocs and in-app VUserManualDialog).
"""

import os
import re
import pytest

DOC_DIRS = [
    os.path.join("spectroview", "resources", "user_manual"),
    "docs",
]

LIST_ITEM_RE = re.compile(r'^([ \t]*(?:>[ \t]*)*)([-*+]|\d+\.)[ \t]+')
TABLE_HEADER_RE = re.compile(r'^[ \t]*\|.+\|[ \t]*$')
TABLE_DIVIDER_RE = re.compile(r'^[ \t]*\|?[ \t]*[-:]+[-| :]*\|[ \t]*$')
HEADING_RE = re.compile(r'^[ \t]*#+')
ADMONITION_RE = re.compile(r'^[ \t]*!!!')
TAB_RE = re.compile(r'^[ \t]*===')
HTML_CLOSE_RE = re.compile(r'^[ \t]*</[a-zA-Z0-9]+>[ \t]*$')
HR_RE = re.compile(r'^[ \t]*(?:---|\*\*\*|___|_______)[ \t]*$')


def get_all_markdown_files():
    md_files = []
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for d in DOC_DIRS:
        abs_d = os.path.join(project_root, d)
        if not os.path.exists(abs_d):
            continue
        for root, _, files in os.walk(abs_d):
            # Skip generated or transient directories
            if any(part in root for part in ["site", ".venv", "build", "dist"]):
                continue
            for f in files:
                if f.endswith(".md"):
                    rel = os.path.relpath(os.path.join(root, f), project_root).replace('\\', '/')
                    # Exclude auto-copied user_manual in docs/ if present during build
                    if rel.startswith("docs/user_manual/"):
                        continue
                    md_files.append((project_root, rel))
    return md_files


@pytest.mark.parametrize("project_root, rel_path", get_all_markdown_files())
def test_markdown_file_spacing(project_root, rel_path):
    """Verify that every list item and table in documentation files is separated
    from preceding paragraph text by a blank line."""
    full_path = os.path.join(project_root, rel_path)
    with open(full_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_fence = False
    fence_marker = ""
    in_frontmatter = False
    in_list = False
    list_indent = 0

    violations = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Frontmatter
        if i == 0 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue

        # Code fences
        if stripped.startswith("```") or stripped.startswith("~~~"):
            if not in_fence:
                in_fence = True
                fence_marker = stripped[:3]
                in_list = False
            elif stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue

        if in_fence:
            continue

        # Blank line resets list state
        if not stripped:
            in_list = False
            continue

        # Check table directly following non-blank paragraph text
        if i + 1 < len(lines) and TABLE_HEADER_RE.match(line) and TABLE_DIVIDER_RE.match(lines[i + 1]):
            prev = lines[i - 1].strip() if i > 0 else ""
            if (
                prev
                and not HEADING_RE.match(prev)
                and not prev.startswith("```")
                and not prev.startswith("<")
                and not prev.startswith("<!--")
                and not HR_RE.match(prev)
            ):
                violations.append(f"Line {i + 1}: table directly follows paragraph text: '{prev[:50]}'")
            in_list = False
            continue

        # Check list item directly following non-blank paragraph text
        m_list = LIST_ITEM_RE.match(line)
        if m_list:
            prefix = m_list.group(1)
            indent = len(prefix.expandtabs(4))

            if not in_list and i > 0:
                prev = lines[i - 1].strip()
                prev_text = prev.lstrip('> ').strip()
                if (
                    prev_text
                    and not LIST_ITEM_RE.match(lines[i - 1])
                    and not HEADING_RE.match(prev)
                    and not ADMONITION_RE.match(prev)
                    and not TAB_RE.match(prev)
                    and not prev.startswith("```")
                    and not prev.startswith("<!--")
                    and not HTML_CLOSE_RE.match(prev)
                    and not HR_RE.match(prev)
                ):
                    violations.append(f"Line {i + 1}: list item directly follows paragraph text: '{prev[:50]}'")
            in_list = True
            list_indent = indent
            continue

        # Continuation line inside an active list
        if in_list:
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= list_indent and not stripped.startswith(">"):
                in_list = False

    assert not violations, f"Markdown spacing violations in {rel_path}:\n" + "\n".join(violations)
