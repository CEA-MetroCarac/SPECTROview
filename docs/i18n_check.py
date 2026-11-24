"""Multilingual documentation structural and parity validator for SPECTROview.

Provides heading-ID extraction matching MkDocs/Python-Markdown rendering
(including ``docs_hooks.ensure_markdown_spacing``) and structural validation
comparing translated ``.<lang>.md`` pages against their English defaults.

Usage:
    python docs/i18n_check.py slugs <en_file>
    python docs/i18n_check.py check <lang> [rel_paths...]
    python docs/i18n_check.py --report [langs...]
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import markdown
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
SUPPORTED_LOCALES = ("fr", "de", "ru", "zh")
CHECKABLE_LOCALES = ("de", "ru", "zh")

_SITE_LOCALE_URL_RE = re.compile(
    r"^(https?://cea-metrocarac\.github\.io/spectroview)/(?:(?:fr|de|ru|zh)/)?(.*)$",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
_HEADING_LINE_RE = re.compile(r"^[ \t]*(#{1,6})\s+(.+?)\s*$")
_MATERIAL_ICON_SHORTCODE_RE = re.compile(r":(?:material|octicons|fontawesome|simple)-[a-z0-9_-]+:")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_ATTR_LIST_RE = re.compile(r"\{\s*#[^}]+\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_CJK_RE = re.compile(r"[\u4E00-\u9FFF]")


def _load_docs_hooks():
    hooks_path = DOCS_DIR / "docs_hooks.py"
    spec = importlib.util.spec_from_file_location("spectroview_docs_hooks", hooks_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {hooks_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DOCS_HOOKS = _load_docs_hooks()


class _MkDocsYamlLoader(yaml.SafeLoader):
    """Safe YAML loader ignoring ``!!python/name:`` plugin tags."""


_MkDocsYamlLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/",
    lambda loader, suffix, node: f"python/name:{suffix}",
)


def load_nav_paths() -> list[str]:
    """Return all relative Markdown file paths registered in ``mkdocs.yml`` nav."""
    with MKDOCS_YML.open("r", encoding="utf-8") as fh:
        cfg = yaml.load(fh, Loader=_MkDocsYamlLoader)

    def _walk(items) -> list[str]:
        out: list[str] = []
        for item in items:
            if isinstance(item, dict):
                for val in item.values():
                    if isinstance(val, str):
                        out.append(val)
                    elif isinstance(val, list):
                        out.extend(_walk(val))
            elif isinstance(item, str):
                out.append(item)
        return out

    return _walk(cfg.get("nav", []))


def _make_markdown_renderer() -> markdown.Markdown:
    extensions = [
        "toc",
        "attr_list",
        "md_in_html",
        "tables",
        "admonition",
        "pymdownx.details",
        "pymdownx.superfences",
        "pymdownx.tabbed",
        "pymdownx.inlinehilite",
        "pymdownx.critic",
        "pymdownx.caret",
        "pymdownx.mark",
        "pymdownx.tilde",
        "pymdownx.keys",
    ]
    extension_configs: dict = {
        "pymdownx.tabbed": {"alternate_style": True},
        "toc": {"permalink": True, "toc_depth": 4},
    }
    try:
        from material.extensions import emoji as mat_emoji

        extensions.append("pymdownx.emoji")
        extension_configs["pymdownx.emoji"] = {
            "emoji_index": mat_emoji.twemoji,
            "emoji_generator": mat_emoji.to_svg,
        }
    except ImportError:
        pass

    return markdown.Markdown(
        extensions=extensions,
        extension_configs=extension_configs,
    )


class _RenderedHtmlParser(HTMLParser):
    """Collect heading IDs, image sources, and link targets from rendered HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.heading_ids: list[str] = []
        self.img_srcs: list[str] = []
        self.link_hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            hid = attr_map.get("id")
            if hid:
                self.heading_ids.append(hid)
        elif tag == "img":
            src = attr_map.get("src")
            if src:
                self.img_srcs.append(src)
        elif tag == "a":
            href = attr_map.get("href")
            classes = (attr_map.get("class") or "").split()
            if href is not None and "headerlink" not in classes:
                self.link_hrefs.append(href)


def _has_material_emoji() -> bool:
    """Return True when ``material.extensions.emoji`` is importable."""
    try:
        from material.extensions import emoji

        return emoji is not None
    except ImportError:
        return False


def _strip_material_icons_in_headings(md_text: str) -> str:
    """Strip Material icon shortcodes from ATX headings outside fenced code blocks."""
    out: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    for line in md_text.splitlines():
        m_fence = _FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len and not m_fence.group(2).strip():
                in_fence = False
            out.append(line)
            continue
        if not in_fence:
            m_head = _HEADING_LINE_RE.match(line)
            if m_head:
                indent = line[: m_head.start(1)]
                hashes, title = m_head.group(1), m_head.group(2)
                cleaned = _MATERIAL_ICON_SHORTCODE_RE.sub("", title)
                cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
                line = f"{indent}{hashes} {cleaned}" if cleaned else f"{indent}{hashes}"
        out.append(line)
    trailing = "\n" if md_text.endswith("\n") else ""
    return "\n".join(out) + trailing


def render_and_extract(md_text: str) -> tuple[list[str], list[str], list[str]]:
    """Render Markdown and return ``(heading_ids, img_srcs, link_hrefs)``."""
    spaced = _DOCS_HOOKS.ensure_markdown_spacing(md_text)
    if not _has_material_emoji():
        spaced = _strip_material_icons_in_headings(spaced)
    md = _make_markdown_renderer()
    html = md.convert(spaced)
    parser = _RenderedHtmlParser()
    parser.feed(html)
    return parser.heading_ids, parser.img_srcs, parser.link_hrefs


def heading_ids(md_text: str) -> list[str]:
    """Return HTML heading IDs for ``md_text`` in document order."""
    ids, _, _ = render_and_extract(md_text)
    return ids


def extract_markdown_headings(md_text: str) -> list[str]:
    """Return raw ATX heading lines outside fenced code blocks."""
    headings: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    for line in md_text.splitlines():
        m_fence = _FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len and not m_fence.group(2).strip():
                in_fence = False
            continue
        if not in_fence and _HEADING_LINE_RE.match(line):
            headings.append(line.strip())
    return headings


def extract_fenced_blocks(md_text: str) -> list[tuple[str, str]]:
    """Return ``(lang_tag, normalized_body)`` for all fenced code blocks."""
    blocks: list[tuple[str, str]] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    fence_lang = ""
    buf: list[str] = []

    for line in md_text.splitlines():
        m_fence = _FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)
            info = m_fence.group(2).strip()
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                fence_lang = info.split()[0].lower() if info else ""
                buf = []
                continue
            if marker[0] == fence_char and len(marker) >= fence_len and not info:
                in_fence = False
                norm_body = "\n".join(ln.rstrip() for ln in buf).strip()
                blocks.append((fence_lang, norm_body))
                buf = []
                continue
        if in_fence:
            buf.append(line)
    return blocks


def _normalize_link_target(href: str) -> str:
    """Normalize self-hosted SPECTROview locale prefixes for EN vs translated comparison."""
    m = _SITE_LOCALE_URL_RE.match(href.strip())
    if m:
        base, rest = m.groups()
        return f"{base.lower()}/{rest}"
    return href.strip()


def _extract_prose_lines(md_text: str) -> list[str]:
    """Return non-empty prose lines outside code fences, headings, rules, and pure tags."""
    lines = md_text.splitlines()
    out: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    in_frontmatter = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if idx == 0 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue

        m_fence = _FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)
            info = m_fence.group(2).strip()
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                continue
            if marker[0] == fence_char and len(marker) >= fence_len and not info:
                in_fence = False
                continue
        if in_fence:
            continue

        if not stripped:
            continue
        if _HEADING_LINE_RE.match(line):
            continue
        if re.match(r"^(?:---|\*\*\*|___|_______)$", stripped):
            continue
        if re.match(r"^(?:</?[a-zA-Z][^>]*>|<!--.*-->)$", stripped):
            continue
        if re.match(r"^!\[.*\]\(.*\)\s*(?:<br\s*/?>)?$", stripped):
            continue
        if stripped.startswith("$$") and stripped.endswith("$$"):
            continue
        if stripped.startswith("|") and re.match(r"^\|[\s:\-|]+\|$", stripped):
            continue

        # Remove inline code & URLs before checking whether the line holds real prose words
        no_code = _INLINE_CODE_RE.sub(" ", stripped)
        no_code = _URL_RE.sub(" ", no_code)
        no_code = _HTML_TAG_RE.sub(" ", no_code)
        # Skip pure citation line if it starts with > Le, Van Hoan
        if "Le, Van Hoan" in stripped and "10.1016/j.softx.2026.102862" in stripped:
            continue
        # Require at least two alphabetic words outside code/tags/URLs
        words = re.findall(r"[^\W\d_]{2,}", no_code, flags=re.UNICODE)
        if len(words) < 2:
            continue
        out.append(stripped)
    return out


def _prose_text_for_script_check(md_text: str) -> str:
    """Return prose text stripped of code blocks, inline code, URLs, HTML tags, and IDs."""
    lines = _extract_prose_lines(md_text)
    cleaned_lines: list[str] = []
    for ln in lines:
        s = _INLINE_CODE_RE.sub(" ", ln)
        s = _ATTR_LIST_RE.sub(" ", s)
        s = _URL_RE.sub(" ", s)
        s = _HTML_TAG_RE.sub(" ", s)
        cleaned_lines.append(s)
    # Include headings (without {#id} and backticks) so short pages like changelog count too
    for h in extract_markdown_headings(md_text):
        s = _HEADING_LINE_RE.sub(r"\2", h)
        s = _INLINE_CODE_RE.sub(" ", s)
        s = _ATTR_LIST_RE.sub(" ", s)
        cleaned_lines.append(s)
    return "\n".join(cleaned_lines)


def translated_path_for(rel_path: str, lang: str) -> Path:
    """Return the localized Path in ``docs/`` for ``rel_path`` and ``lang``."""
    en_path = DOCS_DIR / rel_path
    return en_path.with_name(f"{en_path.stem}.{lang}.md")


def validate_page(rel_path: str, lang: str) -> list[str]:
    """Validate ``rel_path`` in ``lang`` against its English source.

    Returns a list of human-readable error strings (empty if valid).
    """
    errors: list[str] = []
    en_path = DOCS_DIR / rel_path
    tr_path = translated_path_for(rel_path, lang)

    if not en_path.is_file():
        return [f"Missing English source: {rel_path}"]
    if not tr_path.is_file():
        return [f"Missing translation file: {tr_path.relative_to(REPO_ROOT)}"]
    if tr_path.stat().st_size == 0:
        return [f"Empty translation file: {tr_path.relative_to(REPO_ROOT)}"]

    en_text = en_path.read_text(encoding="utf-8")
    tr_text = tr_path.read_text(encoding="utf-8")

    # (b) Heading IDs must match English exactly
    en_hids, en_imgs, en_links = render_and_extract(en_text)
    tr_hids, tr_imgs, tr_links = render_and_extract(tr_text)
    if tr_hids != en_hids:
        errors.append(
            f"Heading IDs mismatch: expected {en_hids}, got {tr_hids}"
        )

    # (c) Fenced code blocks count and non-mermaid contents must match
    en_fences = extract_fenced_blocks(en_text)
    tr_fences = extract_fenced_blocks(tr_text)
    if len(en_fences) != len(tr_fences):
        errors.append(
            f"Fenced code block count mismatch: expected {len(en_fences)}, got {len(tr_fences)}"
        )
    else:
        for idx, ((el, eb), (tl, tb)) in enumerate(zip(en_fences, tr_fences), start=1):
            if el != tl:
                errors.append(
                    f"Fenced code block #{idx} language mismatch: expected '{el}', got '{tl}'"
                )
            elif el != "mermaid" and eb != tb:
                errors.append(f"Fenced code block #{idx} content modified")

    # (d) Multiset of image paths must match
    en_img_counts = Counter(en_imgs)
    tr_img_counts = Counter(tr_imgs)
    if en_img_counts != tr_img_counts:
        missing_imgs = list((en_img_counts - tr_img_counts).elements())
        extra_imgs = list((tr_img_counts - en_img_counts).elements())
        errors.append(
            f"Image paths mismatch: missing={missing_imgs}, extra={extra_imgs}"
        )

    # (e) Set of link targets must match (normalizing self-hosted locale prefix)
    en_link_set = {_normalize_link_target(h) for h in en_links}
    tr_link_set = {_normalize_link_target(h) for h in tr_links}
    if en_link_set != tr_link_set:
        missing_links = sorted(en_link_set - tr_link_set)
        extra_links = sorted(tr_link_set - en_link_set)
        errors.append(
            f"Link targets mismatch: missing={missing_links}, extra={extra_links}"
        )

    # Ensure self-hosted links in translated docs point to /<lang>/ and not another locale
    wrong_locale_links = []
    for h in tr_links:
        m = _SITE_LOCALE_URL_RE.match(h.strip())
        if m:
            expected_prefix = f"/{lang}/"
            if expected_prefix not in h.lower() and not h.lower().endswith(f"/{lang}"):
                wrong_locale_links.append(h)
    if wrong_locale_links:
        errors.append(
            f"Self-site links must use '/{lang}/' prefix: {wrong_locale_links}"
        )

    # (f) Untranslated prose check
    en_prose = set(_extract_prose_lines(en_text))
    tr_prose = _extract_prose_lines(tr_text)
    if tr_prose:
        identical = [ln for ln in tr_prose if ln in en_prose]
        ratio = len(identical) / len(tr_prose)
        if ratio > 0.15:
            errors.append(
                f"Too many untranslated English prose lines ({len(identical)}/{len(tr_prose)} = {ratio:.0%}, max 15%): e.g. {identical[:3]}"
            )

    # (g) Script check for Russian and Simplified Chinese
    script_sample = _prose_text_for_script_check(tr_text)
    if lang == "ru":
        letters = [ch for ch in script_sample if ch.isalpha()]
        cyr = _CYRILLIC_RE.findall(script_sample)
        if letters and (len(cyr) / len(letters)) < 0.30:
            errors.append(
                f"Russian Cyrillic letter ratio too low: {len(cyr)}/{len(letters)} = {len(cyr)/len(letters):.0%} (<30%)"
            )
    elif lang == "zh":
        non_ws = [ch for ch in script_sample if not ch.isspace()]
        cjk = _CJK_RE.findall(script_sample)
        if non_ws and (len(cjk) / len(non_ws)) < 0.20:
            errors.append(
                f"Chinese CJK character ratio too low: {len(cjk)}/{len(non_ws)} = {len(cjk)/len(non_ws):.0%} (<20%)"
            )

    # (h) Locale timestamp format check on user_manual/index
    if rel_path == "user_manual/index.md":
        pattern = _DOCS_HOOKS._STAMP_SPECS[lang][0]
        if not pattern.search(tr_text):
            errors.append(f"Missing or malformed locale timestamp stamp in {tr_path.name}")

    return errors


def validate_language(lang: str, rel_paths: Iterable[str] | None = None) -> dict[str, list[str]]:
    """Validate all or selected nav pages for ``lang``. Return failing ``{rel_path: errors}``."""
    targets = list(rel_paths) if rel_paths else load_nav_paths()
    failures: dict[str, list[str]] = {}
    for rel in targets:
        errs = validate_page(rel, lang)
        if errs:
            failures[rel] = errs
    return failures


def cmd_slugs(en_file: str) -> int:
    path = Path(en_file)
    if not path.is_absolute():
        candidate = REPO_ROOT / en_file
        path = candidate if candidate.is_file() else (DOCS_DIR / en_file)
    if not path.is_file():
        print(f"File not found: {en_file}", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")
    md_headings = extract_markdown_headings(text)
    hids = heading_ids(text)
    if len(md_headings) != len(hids):
        print(
            f"Warning: markdown heading count ({len(md_headings)}) != HTML heading ID count ({len(hids)})",
            file=sys.stderr,
        )
    for h_line, hid in zip(md_headings, hids):
        print(f"{hid}\t{h_line}")
    if len(hids) > len(md_headings):
        for hid in hids[len(md_headings) :]:
            print(f"{hid}\t<html-heading>")
    return 0


def cmd_check(lang: str, rel_paths: list[str]) -> int:
    if lang not in CHECKABLE_LOCALES:
        print(f"Unsupported check locale '{lang}'. Choose from {CHECKABLE_LOCALES}.", file=sys.stderr)
        return 2

    targets = rel_paths or load_nav_paths()
    failures = validate_language(lang, targets)
    passed = len(targets) - len(failures)
    print(f"[i18n_check] {lang}: {passed}/{len(targets)} pages passed")
    for rel, errs in failures.items():
        print(f"  FAIL {rel}:")
        for e in errs:
            print(f"    - {e}")
    return 1 if failures else 0


def cmd_report(langs: list[str]) -> int:
    targets = load_nav_paths()
    selected = langs or list(SUPPORTED_LOCALES)
    for lang in selected:
        missing = [
            rel for rel in targets
            if not translated_path_for(rel, lang).is_file()
            or translated_path_for(rel, lang).stat().st_size == 0
        ]
        present = len(targets) - len(missing)
        print(f"{lang}: {present}/{len(targets)} present")
        for rel in missing:
            print(f"  missing: {rel}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SPECTROview multilingual docs validator")
    sub = parser.add_subparsers(dest="command")

    p_slugs = sub.add_parser("slugs", help="Print heading IDs and lines for an English file")
    p_slugs.add_argument("en_file", help="Path to English Markdown file")

    p_check = sub.add_parser("check", help="Validate translated pages against English source")
    p_check.add_argument("lang", help="Target locale (de, ru, zh)")
    p_check.add_argument("rel_paths", nargs="*", help="Optional relative nav paths (e.g. user_manual/graphs.md)")

    parser.add_argument("--report", nargs="*", metavar="LANG", help="Report missing pages per language")

    args = parser.parse_args(argv)
    if args.report is not None:
        return cmd_report(args.report)
    if args.command == "slugs":
        return cmd_slugs(args.en_file)
    if args.command == "check":
        return cmd_check(args.lang, args.rel_paths)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
