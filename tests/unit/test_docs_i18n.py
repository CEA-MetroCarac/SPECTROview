"""
Validation tests for SPECTROview's French–English multilingual MkDocs setup.

Tier 1 tests validate mkdocs.yml i18n configuration, EN/FR file and nav parity,
and docs_hooks.py behavior using only stdlib + PyYAML (so CI remains fast).
Tier 2 builds the full site with ``mkdocs build --strict`` when MkDocs and
``mkdocs-static-i18n`` are installed and checks every built HTML page, image,
internal link, `#fragment` anchor, search index, and language switcher target.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import os
import posixpath
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
SITE_SUBPATH = "/SPECTROview/"


class _MkDocsYamlLoader(yaml.SafeLoader):
    """Safe YAML loader that ignores ``!!python/name:`` plugin tags."""


def _ignore_python_tag(loader: yaml.Loader, suffix: str, node: yaml.Node):
    return f"python/name:{suffix}"


_MkDocsYamlLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/", _ignore_python_tag
)


def _load_mkdocs_config() -> dict:
    with MKDOCS_YML.open("r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_MkDocsYamlLoader)


def _get_i18n_plugin_config(cfg: dict) -> dict:
    for entry in cfg.get("plugins", []):
        if isinstance(entry, dict) and "i18n" in entry:
            return entry["i18n"]
    raise AssertionError("i18n plugin configuration missing from mkdocs.yml")


def _iter_nav_entries(nav_items) -> list[tuple[str, str]]:
    """Return ``(title, rel_md_path_or_none)`` pairs across the nested nav."""
    collected: list[tuple[str, str | None]] = []
    for item in nav_items:
        if isinstance(item, dict):
            for title, value in item.items():
                if isinstance(value, str):
                    collected.append((title, value))
                elif isinstance(value, list):
                    collected.append((title, None))
                    collected.extend(_iter_nav_entries(value))
        elif isinstance(item, str):
            collected.append((item, item))
    return collected


def _load_docs_hooks_module():
    hooks_path = DOCS_DIR / "docs_hooks.py"
    spec = importlib.util.spec_from_file_location("spectroview_docs_hooks", hooks_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mkdocs_i18n_config_valid():
    cfg = _load_mkdocs_config()
    theme = cfg["theme"]
    assert theme["language"] == "en"
    assert theme.get("icon", {}).get("alternate") == "material/web"

    features = theme.get("features", [])
    assert "navigation.instant" not in features
    assert "navigation.instant.progress" not in features

    plugin_names = [
        p if isinstance(p, str) else next(iter(p.keys()))
        for p in cfg.get("plugins", [])
    ]
    assert "search" in plugin_names
    assert "i18n" in plugin_names

    i18n = _get_i18n_plugin_config(cfg)
    assert i18n["docs_structure"] == "suffix"
    assert i18n["fallback_to_default"] is True
    assert i18n["reconfigure_material"] is True
    assert i18n["reconfigure_search"] is True

    langs = {lang["locale"]: lang for lang in i18n["languages"]}
    assert langs["en"]["default"] is True
    assert langs["en"]["build"] is True
    assert langs["fr"]["name"] == "Français"
    assert langs["fr"]["build"] is True


def test_all_nav_pages_have_english_and_french_files():
    cfg = _load_mkdocs_config()
    entries = _iter_nav_entries(cfg["nav"])
    md_paths = [path for _, path in entries if path is not None]
    assert len(md_paths) == 31

    missing: list[str] = []
    empty: list[str] = []
    for rel_path in md_paths:
        en_file = DOCS_DIR / rel_path
        fr_file = en_file.with_name(f"{en_file.stem}.fr.md")
        if not en_file.is_file():
            missing.append(str(en_file.relative_to(REPO_ROOT)))
        elif en_file.stat().st_size == 0:
            empty.append(str(en_file.relative_to(REPO_ROOT)))
        if not fr_file.is_file():
            missing.append(str(fr_file.relative_to(REPO_ROOT)))
        elif fr_file.stat().st_size == 0:
            empty.append(str(fr_file.relative_to(REPO_ROOT)))

    assert not missing, f"Missing nav documentation files: {missing}"
    assert not empty, f"Empty nav documentation files: {empty}"


def test_all_nav_labels_have_french_translations():
    cfg = _load_mkdocs_config()
    i18n = _get_i18n_plugin_config(cfg)
    langs = {lang["locale"]: lang for lang in i18n["languages"]}
    nav_translations = langs["fr"]["nav_translations"]

    titles = {title for title, _ in _iter_nav_entries(cfg["nav"])}
    missing = sorted(t for t in titles if not str(nav_translations.get(t, "")).strip())
    assert not missing, f"Missing French nav_translations for: {missing}"


def test_docs_hooks_date_formatting_en_and_fr(monkeypatch, tmp_path):
    hooks = _load_docs_hooks_module()
    fixed_now = datetime.datetime(2026, 2, 5, 14, 7)

    class _FixedDateTime:
        @staticmethod
        def now():
            return fixed_now

    monkeypatch.setattr(datetime, "datetime", _FixedDateTime)

    en_page = SimpleNamespace(file=SimpleNamespace(src_path="user_manual/index.md"))
    fr_page = SimpleNamespace(file=SimpleNamespace(src_path="user_manual/index.fr.md"))

    en_out = hooks.on_page_markdown(
        "<i>(Last updated: 01 January 2020 at 00:00)</i>", en_page, {}, None
    )
    fr_out = hooks.on_page_markdown(
        "<i>(Dernière mise à jour : 01 janvier 2020 à 00:00)</i>", fr_page, {}, None
    )
    assert "(Last updated: 05 February 2026 at 14:07)" in en_out
    assert "(Dernière mise à jour : 05 février 2026 à 14:07)" in fr_out

    # SPECTROVIEW_DOCS_NO_STAMP=1 must prevent mutating files on disk
    en_idx = tmp_path / "index.md"
    fr_idx = tmp_path / "index.fr.md"
    en_idx.write_text("(Last updated: 01 January 2020 at 00:00)", encoding="utf-8")
    fr_idx.write_text("(Dernière mise à jour : 01 janvier 2020 à 00:00)", encoding="utf-8")
    monkeypatch.setenv("SPECTROVIEW_DOCS_NO_STAMP", "1")
    hooks._update_last_updated_date(str(tmp_path))
    assert "01 January 2020" in en_idx.read_text(encoding="utf-8")
    assert "01 janvier 2020" in fr_idx.read_text(encoding="utf-8")


def test_docs_hooks_rewrites_html_links_and_images_en_and_fr():
    hooks = _load_docs_hooks_module()
    static_files = [
        SimpleNamespace(
            src_path="user_manual_images/Home/logo.png",
            dest_uri="user_manual_images/Home/logo.png",
        )
    ]
    sample_html = (
        '<a href="graphs.md#sec">G</a> '
        '<a href="index.md">Home</a> '
        '<img src="../user_manual_images/Home/logo.png">'
    )

    cases = [
        (
            "user_manual/index.md",
            "user_manual/index.html",
            'href="graphs/#sec"',
            'href="./"',
            'src="../user_manual_images/Home/logo.png"',
        ),
        (
            "user_manual/graphs.md",
            "user_manual/graphs/index.html",
            'href="../graphs/#sec"',
            'href="../"',
            'src="../../user_manual_images/Home/logo.png"',
        ),
        (
            "user_manual/index.fr.md",
            "fr/user_manual/index.html",
            'href="graphs/#sec"',
            'href="./"',
            'src="../../user_manual_images/Home/logo.png"',
        ),
        (
            "user_manual/graphs.fr.md",
            "fr/user_manual/graphs/index.html",
            'href="../graphs/#sec"',
            'href="../"',
            'src="../../../user_manual_images/Home/logo.png"',
        ),
    ]

    for src_path, dest_uri, exp_graphs, exp_index, exp_img in cases:
        page = SimpleNamespace(file=SimpleNamespace(src_path=src_path, dest_uri=dest_uri))
        out = hooks.on_page_content(sample_html, page, {}, static_files)
        assert exp_graphs in out
        assert exp_index in out
        assert exp_img in out


class _HtmlLinkAndIdParser(HTMLParser):
    """Collect element IDs, images, internal links, hero links, and alternates."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.html_lang: str | None = None
        self.ids: set[str] = set()
        self.img_srcs: list[str] = []
        self.content_hrefs: list[str] = []
        self.alternates: list[tuple[str, str]] = []
        self.current_alt_hreflangs: list[str] = []
        self._in_content = False
        self._in_hero = False
        self._in_alt_link = False
        self._alt_href = ""
        self._alt_text_parts: list[str] = []
        self._content_depth = 0
        self._hero_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "html" and "lang" in attr_map:
            self.html_lang = attr_map["lang"]
        if attr_map.get("id"):
            self.ids.add(attr_map["id"])

        classes = (attr_map.get("class") or "").split()
        if tag == "article" and "md-content__inner" in classes:
            self._in_content = True
            self._content_depth = 1
        elif self._in_content:
            self._content_depth += 1

        if tag == "section" and "tx-hero" in classes:
            self._in_hero = True
            self._hero_depth = 1
        elif self._in_hero:
            self._hero_depth += 1

        if tag == "img" and attr_map.get("src"):
            self.img_srcs.append(attr_map["src"])

        if tag == "a" and attr_map.get("href") is not None:
            href = attr_map["href"]
            if "md-select__link" in classes:
                self._in_alt_link = True
                self._alt_href = href
                self._alt_text_parts = []
                if attr_map.get("aria-current") == "true":
                    self.current_alt_hreflangs.append(attr_map.get("hreflang", ""))
            elif self._in_content or self._in_hero:
                self.content_hrefs.append(href)

    def handle_endtag(self, tag: str) -> None:
        if self._in_alt_link and tag == "a":
            self._in_alt_link = False
            self.alternates.append((self._alt_href, "".join(self._alt_text_parts).strip()))
        if self._in_content:
            self._content_depth -= 1
            if self._content_depth <= 0:
                self._in_content = False
        if self._in_hero:
            self._hero_depth -= 1
            if self._hero_depth <= 0:
                self._in_hero = False

    def handle_data(self, data: str) -> None:
        if self._in_alt_link:
            self._alt_text_parts.append(data)


def _resolve_site_target(
    site_dir: Path, source_html: Path, href: str
) -> tuple[Path, str] | None:
    """Resolve a local HTML href/src relative to ``source_html`` inside ``site_dir``."""
    parts = urlsplit(href)
    if parts.scheme in ("http", "https", "mailto", "tel", "javascript", "data"):
        return None

    fragment = unquote(parts.fragment)
    raw_path = unquote(parts.path)
    if not raw_path:
        return source_html, fragment

    if raw_path.startswith(SITE_SUBPATH):
        rel_from_root = raw_path[len(SITE_SUBPATH) :]
        candidate = site_dir / rel_from_root
    elif raw_path.startswith("/"):
        candidate = site_dir / raw_path.lstrip("/")
    else:
        src_rel = source_html.relative_to(site_dir).as_posix()
        joined = posixpath.normpath(posixpath.join(posixpath.dirname(src_rel), raw_path))
        candidate = site_dir / joined

    if candidate.is_dir():
        candidate = candidate / "index.html"
    elif not candidate.exists() and not candidate.suffix:
        idx_candidate = candidate / "index.html"
        if idx_candidate.exists():
            candidate = idx_candidate

    return candidate, fragment


def test_mkdocs_strict_build_links_images_anchors_and_switcher(tmp_path):
    pytest.importorskip("mkdocs")
    pytest.importorskip("mkdocs_static_i18n")

    site_dir = tmp_path / "site"
    env = os.environ.copy()
    env["SPECTROVIEW_DOCS_NO_STAMP"] = "1"
    env["NO_MKDOCS_2_WARNING"] = "1"

    proc = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-d", str(site_dir)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, f"mkdocs build --strict failed:\n{combined}"

    # Verify root landing pages, hero CTA localization, and EN/FR language switcher badge
    en_home_html = (site_dir / "index.html").read_text(encoding="utf-8")
    fr_home_html = (site_dir / "fr" / "index.html").read_text(encoding="utf-8")
    assert "Get Started" in en_home_html
    assert "Commencer" in fr_home_html
    assert '<span class="md-lang-switcher__code">EN</span>' in en_home_html
    assert '<span class="md-lang-switcher__code">FR</span>' in fr_home_html
    assert "m12.87 15.07-2.54-2.51" not in en_home_html
    assert "m12.87 15.07-2.54-2.51" not in fr_home_html

    # Verify search index covers both EN and FR
    search_index_path = site_dir / "search" / "search_index.json"
    assert search_index_path.is_file()
    search_data = json.loads(search_index_path.read_text(encoding="utf-8"))
    assert search_data["config"]["lang"] == ["en", "fr"]
    locations = {doc["location"] for doc in search_data["docs"]}
    assert any(loc.startswith("fr/") for loc in locations)
    assert any(loc.startswith("user_manual/") for loc in locations)

    # Verify all 31 nav pages are built for both EN and FR
    cfg = _load_mkdocs_config()
    for _, rel_md in _iter_nav_entries(cfg["nav"]):
        if rel_md is None:
            continue
        stem_dir = rel_md[:-3]  # strip .md
        if stem_dir == "index":
            en_target = site_dir / "index.html"
            fr_target = site_dir / "fr" / "index.html"
        elif stem_dir.endswith("/index"):
            en_target = site_dir / stem_dir[:-5] / "index.html"
            fr_target = site_dir / "fr" / stem_dir[:-5] / "index.html"
        else:
            en_target = site_dir / stem_dir / "index.html"
            fr_target = site_dir / "fr" / stem_dir / "index.html"
        assert en_target.is_file(), f"Missing built EN page for {rel_md}"
        assert fr_target.is_file(), f"Missing built FR page for {rel_md}"

    # Parse every HTML file to verify images, internal links, anchors, and switchers
    parsed_pages: dict[Path, _HtmlLinkAndIdParser] = {}
    for html_file in site_dir.rglob("*.html"):
        rel_parts = html_file.relative_to(site_dir).parts
        if html_file.name == "404.html" or rel_parts[0] == "overrides":
            continue
        parser = _HtmlLinkAndIdParser()
        parser.feed(html_file.read_text(encoding="utf-8"))
        parsed_pages[html_file.resolve()] = parser

    broken_images: list[str] = []
    broken_links: list[str] = []
    broken_anchors: list[str] = []

    for html_file, parser in parsed_pages.items():
        rel_src = html_file.relative_to(site_dir).as_posix()
        for src in parser.img_srcs:
            resolved = _resolve_site_target(site_dir, html_file, src)
            if resolved is None:
                continue
            target_path, _ = resolved
            if not target_path.is_file():
                broken_images.append(f"{rel_src} -> {src}")

        for href in parser.content_hrefs:
            resolved = _resolve_site_target(site_dir, html_file, href)
            if resolved is None:
                continue
            target_path, fragment = resolved
            resolved_path = target_path.resolve()
            if not resolved_path.is_file():
                broken_links.append(f"{rel_src} -> {href}")
                continue
            if fragment:
                target_parser = parsed_pages.get(resolved_path)
                if target_parser is None or fragment not in target_parser.ids:
                    broken_anchors.append(f"{rel_src} -> {href} (#{fragment})")

    assert not broken_images, "Broken images in built site:\n" + "\n".join(broken_images)
    assert not broken_links, "Broken links in built site:\n" + "\n".join(broken_links)
    assert not broken_anchors, "Broken anchors in built site:\n" + "\n".join(broken_anchors)

    # Verify per-page language switcher links across key EN & FR pages
    switcher_pairs = [
        ("index.html", "fr/index.html", "en", "fr"),
        ("user_manual/graphs/index.html", "fr/user_manual/graphs/index.html", "en", "fr"),
        ("api/spectra/index.html", "fr/api/spectra/index.html", "en", "fr"),
        ("developer/vbf_engine/index.html", "fr/developer/vbf_engine/index.html", "en", "fr"),
    ]
    for en_rel, fr_rel, exp_en_lang, exp_fr_lang in switcher_pairs:
        en_file = (site_dir / en_rel).resolve()
        fr_file = (site_dir / fr_rel).resolve()
        en_parser = parsed_pages[en_file]
        fr_parser = parsed_pages[fr_file]
        assert en_parser.html_lang == exp_en_lang
        assert fr_parser.html_lang == exp_fr_lang

        en_alt_map = {label: href for href, label in en_parser.alternates}
        fr_alt_map = {label: href for href, label in fr_parser.alternates}
        assert set(en_alt_map.keys()) == {"English", "Français"}
        assert set(fr_alt_map.keys()) == {"English", "Français"}
        assert en_parser.current_alt_hreflangs == [exp_en_lang]
        assert fr_parser.current_alt_hreflangs == [exp_fr_lang]

        to_fr = _resolve_site_target(site_dir, en_file, en_alt_map["Français"])
        to_en = _resolve_site_target(site_dir, fr_file, fr_alt_map["English"])
        assert to_fr is not None and to_fr[0].resolve() == fr_file
        assert to_en is not None and to_en[0].resolve() == en_file
