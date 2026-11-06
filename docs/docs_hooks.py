import os
import re
import sys

# Track whether we are running in persistent live server mode ('mkdocs serve')
_is_serve_mode: bool = any(arg in sys.argv for arg in ("serve",))


def on_startup(command, dirty):
    global _is_serve_mode
    if command == "serve":
        _is_serve_mode = True


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

    This prevents Python-Markdown (used by MkDocs) from collapsing bullet
    and numbered lists or tables into the preceding paragraph as plain text.
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
        if stripped.startswith('```') or stripped.startswith('~~~'):
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

            if not in_list:
                if out:
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


def on_page_content(html, page, config, files):
    """
    Hook to fix raw HTML links in the generated HTML content.

    MkDocs normally doesn't rewrite links inside raw HTML <a> tags.
    This hook finds relative href="filename.md" links in the user manual
    and rewrites them to href="filename/" so they match MkDocs'
    directory-style URL structure.
    """
    src_path = page.file.src_path.replace('\\', '/')
    if src_path.startswith("user_manual/"):
        # Determine if we are at the user_manual/ root (index.md)
        # or in a sub-directory (e.g. user_manual/01_introduction/)
        is_index = src_path.endswith("index.md")
        prefix = "" if is_index else "../"

        def replace_link(match):
            filename = match.group(1)
            anchor = match.group(2)
            # Special case for index.md: link to the directory root
            if filename == "index":
                return f'href="{prefix or "./"}{anchor}"'
            # General case: link to the file's directory URL
            return f'href="{prefix}{filename}/{anchor}"'

        # Match href="filename.md" or href="filename.md#anchor"
        # but ignore external (http) or absolute (/) links.
        html = re.sub(
            r'href="(?!(?:https?://|/))([^"#]+)\.md(#?[^"]*)"',
            replace_link,
            html
        )

        # Fix image src attributes inside HTML tags.
        # MkDocs automatically fixes relative paths for Markdown images ![alt](path),
        # but leaves raw HTML <img src="..."> untouched.
        # From site/user_manual/index.html (is_index=True), "../user_manual_images/" is correct.
        # From site/user_manual/page_name/index.html (is_index=False), we need "../../user_manual_images/"
        if not is_index:
            html = re.sub(
                r'src="\.\./user_manual_images/',
                r'src="../../user_manual_images/',
                html
            )
    return html


def on_page_markdown(markdown, page, config, files):
    """
    Hook to dynamically ensure the rendered user manual index.md always has the current date,
    and guarantee that lists and tables are properly spaced for Python-Markdown.
    """
    src_path = page.file.src_path.replace('\\', '/')
    if src_path.endswith("index.md") and "user_manual" in src_path:
        import datetime
        timestamp_str = datetime.datetime.now().strftime("%d %B %Y at %H:%M")
        markdown = re.sub(
            r"\(Last updated:\s*[^)]+\)",
            f"(Last updated: {timestamp_str})",
            markdown,
            flags=re.IGNORECASE
        )
    return ensure_markdown_spacing(markdown)


def on_pre_build(config, **kwargs):
    """
    Hook that runs before the MkDocs build process starts.
    Updates '(Last updated: DD Month YYYY at HH:MM)' in docs/user_manual/index.md
    when building or deploying (skipped during 'mkdocs serve' to avoid rebuild loops).
    """
    docs_dir = config['docs_dir']
    manual_dir = os.path.join(docs_dir, "user_manual")
    _update_last_updated_date(manual_dir)


def _update_last_updated_date(manual_dir: str) -> None:
    """Automatically update '(Last updated: DD Month YYYY at HH:MM)' in user_manual/index.md."""
    if _is_serve_mode:
        return
    import datetime
    index_path = os.path.join(manual_dir, "index.md")
    if not os.path.isfile(index_path):
        return

    timestamp_str = datetime.datetime.now().strftime("%d %B %Y at %H:%M")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = r"\(Last updated:\s*[^)]+\)"
        new_text = f"(Last updated: {timestamp_str})"
        new_content, count = re.subn(pattern, new_text, content, flags=re.IGNORECASE)

        if count > 0 and new_content != content:
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"  [docs_hook] Automatically updated user manual date and time to: {timestamp_str}")
    except Exception as e:
        print(f"  [docs_hook] Warning: Could not update last updated date: {e}")
