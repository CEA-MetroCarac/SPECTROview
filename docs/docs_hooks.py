import datetime
import os
import posixpath
import re
import sys

# Track whether we are running in persistent live server mode ('mkdocs serve')
_is_serve_mode: bool = any(arg in sys.argv for arg in ("serve",))

_EN_MONTHS = (
    "",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_FR_MONTHS = (
    "",
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)

_INDEX_FILE_RE = re.compile(r'(?:^|/)index(?:\.[a-z]{2}(?:-[A-Z]{2})?)?\.md$')
_EN_STAMP_RE = re.compile(r"\(Last updated:\s*[^)]+\)", re.IGNORECASE)
_FR_STAMP_RE = re.compile(r"\(Dernière mise à jour\s*:\s*[^)]+\)", re.IGNORECASE)


def _format_timestamp(locale: str, now: datetime.datetime | None = None) -> str:
    """Format timestamp using locale-independent English or French month names."""
    if now is None:
        now = datetime.datetime.now()
    if locale == "fr":
        return f"{now.day:02d} {_FR_MONTHS[now.month]} {now.year} à {now.hour:02d}:{now.minute:02d}"
    return f"{now.day:02d} {_EN_MONTHS[now.month]} {now.year} at {now.hour:02d}:{now.minute:02d}"


def _detect_locale(src_path: str) -> str:
    """Infer locale from a suffix-named Markdown source path."""
    norm = src_path.replace("\\", "/")
    if norm.endswith(".fr.md"):
        return "fr"
    return "en"


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


def _resolve_user_manual_images_prefix(page, files, is_index: bool) -> str:
    """Compute the relative prefix to user_manual_images/ from the rendered HTML page."""
    page_dir = posixpath.dirname((page.file.dest_uri or "").replace("\\", "/"))
    if files is not None:
        for f in files:
            src = (getattr(f, "src_path", "") or "").replace("\\", "/")
            if src.startswith("user_manual_images/"):
                dest_uri = (getattr(f, "dest_uri", "") or "").replace("\\", "/")
                idx = dest_uri.find("user_manual_images/")
                if idx != -1:
                    images_root = dest_uri[: idx + len("user_manual_images")]
                    rel = posixpath.relpath(images_root, page_dir or ".")
                    return rel.rstrip("/") + "/"
                break
    return "../user_manual_images/" if is_index else "../../user_manual_images/"


def on_page_content(html, page, config, files):
    """
    Hook to fix raw HTML links in the generated HTML content.

    MkDocs normally doesn't rewrite links inside raw HTML <a> tags.
    This hook finds relative href="filename.md" links in the user manual
    and rewrites them to href="filename/" so they match MkDocs'
    directory-style URL structure, and ensures raw <img> tags pointing
    to user_manual_images/ resolve accurately across locales.
    """
    src_path = page.file.src_path.replace('\\', '/')
    if src_path.startswith("user_manual/") or "/user_manual/" in f"/{src_path}":
        # Determine if we are at the user_manual/ root (index.md / index.fr.md)
        # or in a sub-directory (e.g. user_manual/graphs/)
        is_index = bool(_INDEX_FILE_RE.search(src_path))
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
        img_prefix = _resolve_user_manual_images_prefix(page, files, is_index)
        html = re.sub(
            r'src="(?:\.\./)+user_manual_images/',
            f'src="{img_prefix}',
            html
        )
    return html


def on_page_markdown(markdown, page, config, files):
    """
    Hook to dynamically ensure the rendered user manual index pages (EN and FR)
    always carry the current date, and guarantee that lists and tables are
    properly spaced for Python-Markdown.
    """
    src_path = page.file.src_path.replace('\\', '/')
    if _INDEX_FILE_RE.search(src_path) and "user_manual" in src_path:
        locale = _detect_locale(src_path)
        timestamp_str = _format_timestamp(locale)
        if locale == "fr":
            markdown = _FR_STAMP_RE.sub(
                f"(Dernière mise à jour : {timestamp_str})",
                markdown,
            )
        else:
            markdown = _EN_STAMP_RE.sub(
                f"(Last updated: {timestamp_str})",
                markdown,
            )
    return ensure_markdown_spacing(markdown)


def on_pre_build(config, **kwargs):
    """
    Hook that runs before the MkDocs build process starts.
    Updates timestamps in docs/user_manual/index.md and index.fr.md
    when building or deploying (skipped during 'mkdocs serve' or when
    SPECTROVIEW_DOCS_NO_STAMP=1 is set).
    """
    docs_dir = config['docs_dir']
    manual_dir = os.path.join(docs_dir, "user_manual")
    _update_last_updated_date(manual_dir)


def _update_last_updated_date(manual_dir: str) -> None:
    """Automatically update the Last updated timestamp in user_manual/index[.fr].md."""
    if _is_serve_mode or os.environ.get("SPECTROVIEW_DOCS_NO_STAMP") == "1":
        return

    now = datetime.datetime.now()
    targets = [
        ("index.md", "en", _EN_STAMP_RE, f"(Last updated: {_format_timestamp('en', now)})"),
        ("index.fr.md", "fr", _FR_STAMP_RE, f"(Dernière mise à jour : {_format_timestamp('fr', now)})"),
    ]
    for filename, locale, pattern, new_text in targets:
        index_path = os.path.join(manual_dir, filename)
        if not os.path.isfile(index_path):
            continue
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                content = f.read()

            new_content, count = pattern.subn(new_text, content)
            if count > 0 and new_content != content:
                with open(index_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(
                    f"  [docs_hook] Automatically updated {filename} ({locale}) "
                    f"date and time to: {_format_timestamp(locale, now)}"
                )
        except OSError as e:
            print(f"  [docs_hook] Warning: Could not update {filename} last updated date: {e}")
