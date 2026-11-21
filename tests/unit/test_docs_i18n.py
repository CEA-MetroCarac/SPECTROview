"""
Validation tests for SPECTROview's multilingual MkDocs setup (EN / FR / DE / RU / ZH).

Tier 1 tests validate mkdocs.yml i18n configuration, file and nav parity,
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


def _load_i18n_check_module():
    chk_path = DOCS_DIR / "i18n_check.py"
    spec = importlib.util.spec_from_file_location("spectroview_i18n_check", chk_path)
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
    expected_names = {
        "fr": "Français",
        "de": "Deutsch",
        "ru": "Русский",
        "zh": "简体中文",
    }
    for locale, native_name in expected_names.items():
        assert locale in langs, f"Missing locale {locale} in mkdocs.yml"
        assert langs[locale]["name"] == native_name
        assert langs[locale]["build"] is True


@pytest.mark.parametrize("locale", ["fr", "de", "ru", "zh"])
def test_all_nav_pages_have_localized_files(locale: str):
    cfg = _load_mkdocs_config()
    entries = _iter_nav_entries(cfg["nav"])
    md_paths = [path for _, path in entries if path is not None]
    assert len(md_paths) == 31

    missing: list[str] = []
    empty: list[str] = []
    for rel_path in md_paths:
        en_file = DOCS_DIR / rel_path
        loc_file = en_file.with_name(f"{en_file.stem}.{locale}.md")
        if not en_file.is_file():
            missing.append(str(en_file.relative_to(REPO_ROOT)))
        elif en_file.stat().st_size == 0:
            empty.append(str(en_file.relative_to(REPO_ROOT)))
        if not loc_file.is_file():
            missing.append(str(loc_file.relative_to(REPO_ROOT)))
        elif loc_file.stat().st_size == 0:
            empty.append(str(loc_file.relative_to(REPO_ROOT)))

    assert not missing, f"Missing nav documentation files for {locale}: {missing}"
    assert not empty, f"Empty nav documentation files for {locale}: {empty}"


@pytest.mark.parametrize("locale", ["de", "ru", "zh"])
def test_i18n_check_structural_parity(locale: str):
    pytest.importorskip("markdown")
    chk = _load_i18n_check_module()
    failures = chk.validate_language(locale)
    assert not failures, f"Structural i18n_check failures for {locale}: {failures}"


@pytest.mark.parametrize("locale", ["fr", "de", "ru", "zh"])
def test_all_nav_labels_have_translations(locale: str):
    cfg = _load_mkdocs_config()
    i18n = _get_i18n_plugin_config(cfg)
    langs = {lang["locale"]: lang for lang in i18n["languages"]}
    nav_translations = langs[locale]["nav_translations"]

    titles = {title for title, _ in _iter_nav_entries(cfg["nav"])}
    missing = sorted(t for t in titles if not str(nav_translations.get(t, "")).strip())
    assert not missing, f"Missing {locale} nav_translations for: {missing}"


def test_docs_hooks_date_formatting_all_locales(monkeypatch, tmp_path):
    hooks = _load_docs_hooks_module()
    fixed_now = datetime.datetime(2026, 2, 5, 14, 7)

    class _FixedDateTime:
        @staticmethod
        def now():
            return fixed_now

    monkeypatch.setattr(datetime, "datetime", _FixedDateTime)

    cases = [
        ("en", "user_manual/index.md", "(Last updated: 01 January 2020 at 00:00)", "(Last updated: 05 February 2026 at 14:07)"),
        ("fr", "user_manual/index.fr.md", "(Dernière mise à jour : 01 janvier 2020 à 00:00)", "(Dernière mise à jour : 05 février 2026 à 14:07)"),
        ("de", "user_manual/index.de.md", "(Zuletzt aktualisiert: 01. Januar 2020 um 00:00)", "(Zuletzt aktualisiert: 05. Februar 2026 um 14:07)"),
        ("ru", "user_manual/index.ru.md", "(Последнее обновление: 01 января 2020 г. в 00:00)", "(Последнее обновление: 05 февраля 2026 г. в 14:07)"),
        ("zh", "user_manual/index.zh.md", "(最后更新：2020年01月01日 00:00)", "(最后更新：2026年02月05日 14:07)"),
    ]

    for locale, src_path, initial, expected in cases:
        page = SimpleNamespace(file=SimpleNamespace(src_path=src_path))
        out = hooks.on_page_markdown(f"<i>{initial}</i>", page, {}, None)
        assert expected in out, f"Date formatting mismatch for {locale}: {out}"

    # SPECTROVIEW_DOCS_NO_STAMP=1 must prevent mutating files on disk across all locales
    files_to_check = []
    for locale, src_path, initial, _ in cases:
        idx_file = tmp_path / Path(src_path).name
        idx_file.write_text(initial, encoding="utf-8")
        files_to_check.append((idx_file, initial))

    monkeypatch.setenv("SPECTROVIEW_DOCS_NO_STAMP", "1")
    hooks._update_last_updated_date(str(tmp_path))
    for idx_file, initial in files_to_check:
        assert idx_file.read_text(encoding="utf-8") == initial


def test_docs_hooks_rewrites_html_links_and_images_all_locales():
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
    ]
    for loc in ("fr", "de", "ru", "zh"):
        cases.extend(
            [
                (
                    f"user_manual/index.{loc}.md",
                    f"{loc}/user_manual/index.html",
                    'href="graphs/#sec"',
                    'href="./"',
                    'src="../../user_manual_images/Home/logo.png"',
                ),
                (
                    f"user_manual/graphs.{loc}.md",
                    f"{loc}/user_manual/graphs/index.html",
                    'href="../graphs/#sec"',
                    'href="../"',
                    'src="../../../user_manual_images/Home/logo.png"',
                ),
            ]
        )

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

    # Verify root landing pages, hero CTA localization, and language switcher badge across all 5 locales
    home_expectations = [
        ("en", site_dir / "index.html", "Get Started", "EN"),
        ("fr", site_dir / "fr" / "index.html", "Commencer", "FR"),
        ("de", site_dir / "de" / "index.html", "Erste Schritte", "DE"),
        ("ru", site_dir / "ru" / "index.html", "Начать работу", "RU"),
        ("zh", site_dir / "zh" / "index.html", "快速开始", "ZH"),
    ]
    for loc, home_path, expected_cta, expected_badge in home_expectations:
        assert home_path.is_file(), f"Missing home page for {loc}: {home_path}"
        home_html = home_path.read_text(encoding="utf-8")
        assert expected_cta in home_html, f"Missing hero CTA '{expected_cta}' in {loc}"
        assert f'<span class="md-lang-switcher__code">{expected_badge}</span>' in home_html
        assert "m12.87 15.07-2.54-2.51" not in home_html

    # Verify search index covers all 5 languages
    search_index_path = site_dir / "search" / "search_index.json"
    assert search_index_path.is_file()
    search_data = json.loads(search_index_path.read_text(encoding="utf-8"))
    assert search_data["config"]["lang"] == ["en", "fr", "de", "ru", "zh"]
    locations = {doc["location"] for doc in search_data["docs"]}
    for prefix in ("fr/", "de/", "ru/", "zh/", "user_manual/"):
        assert any(loc.startswith(prefix) for loc in locations), f"No search docs under {prefix}"

    # Verify all 31 nav pages are built for all 5 languages
    cfg = _load_mkdocs_config()
    for _, rel_md in _iter_nav_entries(cfg["nav"]):
        if rel_md is None:
            continue
        stem_dir = rel_md[:-3]  # strip .md
        if stem_dir == "index":
            rel_html = Path("index.html")
        elif stem_dir.endswith("/index"):
            rel_html = Path(stem_dir[:-5]) / "index.html"
        else:
            rel_html = Path(stem_dir) / "index.html"

        assert (site_dir / rel_html).is_file(), f"Missing built EN page for {rel_md}"
        for loc in ("fr", "de", "ru", "zh"):
            loc_target = site_dir / loc / rel_html
            assert loc_target.is_file(), f"Missing built {loc.upper()} page for {rel_md}"

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

    # Verify per-page language switcher links across all 5 languages on key pages
    locale_to_label = {
        "en": "English",
        "fr": "Français",
        "de": "Deutsch",
        "ru": "Русский",
        "zh": "简体中文",
    }
    expected_labels = set(locale_to_label.values())
    sample_pages = [
        "index.html",
        "user_manual/graphs/index.html",
        "api/spectra/index.html",
        "developer/vbf_engine/index.html",
    ]
    for sample_rel in sample_pages:
        locale_files = {
            loc: ((site_dir / sample_rel) if loc == "en" else (site_dir / loc / sample_rel)).resolve()
            for loc in locale_to_label
        }
        for src_loc, src_file in locale_files.items():
            src_parser = parsed_pages[src_file]
            assert src_parser.html_lang == src_loc
            assert src_parser.current_alt_hreflangs == [src_loc]
            alt_map = {label: href for href, label in src_parser.alternates}
            assert set(alt_map.keys()) == expected_labels
            for dst_loc, dst_label in locale_to_label.items():
                resolved_target = _resolve_site_target(site_dir, src_file, alt_map[dst_label])
                assert resolved_target is not None
                assert resolved_target[0].resolve() == locale_files[dst_loc]
