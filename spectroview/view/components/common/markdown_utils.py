"""
spectroview/view/components/markdown_utils.py
---------------------------------------------
Markdown preprocessing and parsing utilities.

Addresses Python-Markdown vs CommonMark list parsing discrepancy:
CommonMark (used by VS Code, JetBrains IDEs, and GitHub) allows bullet and
numbered lists to interrupt an active paragraph without a preceding blank line.
Python-Markdown (used by MkDocs and PySide6 QTextBrowser) requires a blank line
before a list; otherwise, list markers are treated as paragraph continuation lines
and collapsed into a single run-on sentence.

This utility safely ensures blank lines precede lists and tables while respecting
fenced code blocks, frontmatter, blockquotes, and list continuation lines.
"""

from __future__ import annotations

import re

# Pre-compiled regex patterns for markdown block elements
_LIST_ITEM_RE = re.compile(r'^([ \t]*(?:>[ \t]*)*)([-*+]|\d+\.)[ \t]+')
_TABLE_ROW_RE = re.compile(r'^[ \t]*\|.+\|[ \t]*$')
_TABLE_DIV_RE = re.compile(r'^[ \t]*\|?[ \t]*[-:]+[-| :]*\|[ \t]*$')
_HEADING_RE = re.compile(r'^[ \t]*#+')
_ADMONITION_RE = re.compile(r'^[ \t]*!!!')
_TAB_RE = re.compile(r'^[ \t]*===')
_HTML_CLOSE_RE = re.compile(r'^[ \t]*</[a-zA-Z0-9]+>[ \t]*$')
_HR_RE = re.compile(r'^[ \t]*(?:---|\*\*\*|___|_______)[ \t]*$')
_BQ_PREFIX_RE = re.compile(r'^([ \t]*(?:>[ \t]*)*)')


def ensure_markdown_spacing(text: str) -> str:
    """Ensure markdown lists and tables that directly follow paragraph text
    have an intervening blank line before them.

    Parameters
    ----------
    text : str
        Raw markdown text.

    Returns
    -------
    str
        Normalized markdown text where all bullet lists, numbered lists,
        and tables following paragraph text are separated by a blank line.
    """
    if not text:
        return text

    lines = text.splitlines(keepends=True)
    out: list[str] = []

    in_fence = False
    fence_marker = ''
    in_frontmatter = False
    in_list = False
    list_indent = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check YAML frontmatter at document start
        if i == 0 and stripped == '---':
            in_frontmatter = True
            out.append(line)
            continue
        if in_frontmatter:
            out.append(line)
            if stripped == '---':
                in_frontmatter = False
            continue

        # Check fenced code blocks (``` or ~~~)
        if stripped.startswith(('```', '~~~')):
            if not in_fence:
                in_fence = True
                fence_marker = stripped[:3]
                in_list = False
            elif stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ''
            out.append(line)
            continue

        if in_fence:
            out.append(line)
            continue

        # Blank line resets active list state and is appended verbatim
        if not stripped:
            in_list = False
            out.append(line)
            continue

        # Check table header followed by divider
        if i + 1 < len(lines) and _TABLE_ROW_RE.match(line) and _TABLE_DIV_RE.match(lines[i + 1]):
            if out:
                prev_line = out[-1]
                prev_stripped = prev_line.strip()
                if (
                    prev_stripped
                    and not _HEADING_RE.match(prev_stripped)
                    and not _ADMONITION_RE.match(prev_stripped)
                    and not _TAB_RE.match(prev_stripped)
                    and not prev_stripped.startswith('```')
                    and not prev_stripped.startswith('<')
                    and not prev_stripped.startswith('<!--')
                    and not _HR_RE.match(prev_stripped)
                ):
                    out.append('\n')
            in_list = False
            out.append(line)
            continue

        # Check list item
        m_list = _LIST_ITEM_RE.match(line)
        if m_list:
            prefix = m_list.group(1)
            indent = len(prefix.expandtabs(4))

            if not in_list and out:
                prev_line = out[-1]
                prev_stripped = prev_line.strip()
                if (
                    prev_stripped
                    and not _HEADING_RE.match(prev_stripped)
                    and not _ADMONITION_RE.match(prev_stripped)
                    and not _TAB_RE.match(prev_stripped)
                    and not prev_stripped.startswith('```')
                    and not prev_stripped.startswith('<!--')
                    and not _HTML_CLOSE_RE.match(prev_stripped)
                    and not _HR_RE.match(prev_stripped)
                ):
                    if '>' in prefix:
                        bq_match = _BQ_PREFIX_RE.match(prefix)
                        bq_prefix = (bq_match.group(1).rstrip() if bq_match else '>') + '\n'
                        out.append(bq_prefix)
                    else:
                        out.append('\n')
            in_list = True
            list_indent = indent
            out.append(line)
            continue

        # If not a list item:
        if in_list:
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= list_indent and not stripped.startswith('>'):
                in_list = False

        out.append(line)

    return ''.join(out)

