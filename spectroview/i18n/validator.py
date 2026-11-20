"""AST-based extraction and validation helpers for SPECTROview translation catalogs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Optional


@dataclass
class ExtractedCatalog:
    """All static translation keys extracted from Python sources via AST."""

    strings: set[str] = field(default_factory=set)
    plurals: set[str] = field(default_factory=set)
    contexts: dict[str, set[str]] = field(default_factory=dict)

    def add_string(self, text: str, ctx: str = "") -> None:
        if not text:
            return
        if ctx:
            self.contexts.setdefault(ctx, set()).add(text)
        else:
            self.strings.add(text)

    def add_plural(self, singular: str, plural: str, ctx: str = "") -> None:
        key = f"{singular}|{plural}"
        if ctx:
            self.contexts.setdefault(ctx, set()).add(key)
        else:
            self.plurals.add(key)


def _str_literal(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _kw_str(call: ast.Call, name: str, default: str = "") -> str:
    for kw in call.keywords:
        if kw.arg == name:
            val = _str_literal(kw.value)
            if val is not None:
                return val
    return default


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _extract_combo_items(node: ast.AST) -> list[str]:
    out: list[str] = []
    if not isinstance(node, (ast.List, ast.Tuple)):
        return out
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            out.append(elt.value)
        elif isinstance(elt, (ast.Tuple, ast.List)) and elt.elts:
            s = _str_literal(elt.elts[0])
            if s is not None:
                out.append(s)
    return out


def extract_keys_from_sources(source_root: Path) -> ExtractedCatalog:
    """Walk all ``*.py`` files under *source_root* and extract translation keys."""
    extracted = ExtractedCatalog()
    for py_file in sorted(source_root.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            ctx = _kw_str(node, "ctx")

            if name == "tr":
                src = _str_literal(node.args[0]) if node.args else _kw_str(node, "text") or None
                if src:
                    extracted.add_string(src, ctx)
            elif name == "tr_plural" and len(node.args) >= 2:
                s1 = _str_literal(node.args[0])
                s2 = _str_literal(node.args[1])
                if s1 and s2:
                    extracted.add_plural(s1, s2, ctx)
            elif name in (
                "bind_text",
                "bind_tooltip",
                "bind_placeholder",
                "bind_title",
                "bind_window_title",
                "bind_form_row",
            ) and len(node.args) >= 2:
                src = _str_literal(node.args[1])
                if src:
                    extracted.add_string(src, ctx)
            elif name in ("add_row", "add_bounds_row") and node.args:
                lbl = _str_literal(node.args[0])
                if lbl:
                    extracted.add_string(lbl, ctx)
                if len(node.args) >= 3:
                    tip = _str_literal(node.args[2])
                    if tip:
                        extracted.add_string(tip, ctx)
            elif name == "add_api_row" and len(node.args) >= 2:
                lbl = _str_literal(node.args[1])
                if lbl:
                    extracted.add_string(lbl, ctx)
                if len(node.args) >= 4:
                    tip = _str_literal(node.args[3])
                    if tip:
                        extracted.add_string(tip, ctx)
            elif name == "_wrap_menu_row" and len(node.args) >= 2:
                lbl = _str_literal(node.args[1])
                if lbl:
                    extracted.add_string(lbl, ctx)
            elif name == "_bind_menu_checkbox" and len(node.args) >= 3:
                txt = _str_literal(node.args[2])
                if txt:
                    extracted.add_string(txt, ctx)
            elif name == "bind_tab_text" and len(node.args) >= 3:
                src = _str_literal(node.args[2])
                if src:
                    extracted.add_string(src, ctx)
            elif name == "bind_plural_text" and len(node.args) >= 3:
                s1 = _str_literal(node.args[1])
                s2 = _str_literal(node.args[2])
                if s1 and s2:
                    extracted.add_plural(s1, s2, ctx)
            elif name == "bind_header_labels" and len(node.args) >= 2:
                seq = node.args[1]
                if isinstance(seq, (ast.List, ast.Tuple)):
                    for elt in seq.elts:
                        s = _str_literal(elt)
                        if s:
                            extracted.add_string(s, ctx)
            elif name == "bind_combo_items" and len(node.args) >= 2:
                for label in _extract_combo_items(node.args[1]):
                    extracted.add_string(label, ctx)
    return extracted


def _placeholder_names(template: str) -> set[str]:
    names: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name is not None:
            names.add(field_name.split(".")[0].split("[")[0])
    return names


@dataclass
class ValidationReport:
    """Result of validating a translation JSON catalog against Python source keys."""

    missing_strings: list[str] = field(default_factory=list)
    empty_strings: list[str] = field(default_factory=list)
    missing_plurals: list[str] = field(default_factory=list)
    empty_plurals: list[str] = field(default_factory=list)
    missing_contexts: list[str] = field(default_factory=list)
    placeholder_mismatches: list[str] = field(default_factory=list)
    unused_strings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.missing_strings
            or self.empty_strings
            or self.missing_plurals
            or self.empty_plurals
            or self.missing_contexts
            or self.placeholder_mismatches
        )


def validate_catalog(
    lang: str = "fr",
    *,
    source_root: Optional[Path] = None,
    catalog_path: Optional[Path] = None,
    required_strings: Iterable[str] = (),
) -> ValidationReport:
    """Validate ``spectroview/resources/i18n/<lang>.json`` against extracted keys."""
    pkg_root = source_root or Path(__file__).resolve().parents[1]
    cat_path = catalog_path or (pkg_root / "resources" / "i18n" / f"{lang}.json")
    catalog = json.loads(cat_path.read_text(encoding="utf-8"))

    extracted = extract_keys_from_sources(pkg_root)
    for req in required_strings:
        extracted.add_string(req)

    strings_map: dict[str, str] = catalog.get("strings", {})
    plurals_map: dict[str, list[str]] = catalog.get("plurals", {})
    contexts_map: dict[str, dict[str, Any]] = catalog.get("contexts", {})

    report = ValidationReport()

    for key in sorted(extracted.strings):
        if key not in strings_map:
            report.missing_strings.append(key)
        else:
            val = strings_map[key]
            if not isinstance(val, str) or not val.strip():
                report.empty_strings.append(key)
            elif _placeholder_names(key) != _placeholder_names(val):
                report.placeholder_mismatches.append(f"{key!r} -> {val!r}")

    for key in sorted(extracted.plurals):
        if key not in plurals_map:
            report.missing_plurals.append(key)
        else:
            forms = plurals_map[key]
            s_en, p_en = key.split("|", 1)
            if (
                not isinstance(forms, list)
                or len(forms) < 2
                or not all(isinstance(f, str) and f.strip() for f in forms[:2])
            ):
                report.empty_plurals.append(key)
            else:
                expected = _placeholder_names(s_en) | _placeholder_names(p_en)
                if not all(_placeholder_names(f) <= expected for f in forms[:2]):
                    report.placeholder_mismatches.append(f"{key!r} -> {forms!r}")

    for ctx, keys in sorted(extracted.contexts.items()):
        ctx_dict = contexts_map.get(ctx, {})
        for key in sorted(keys):
            if key not in ctx_dict or not ctx_dict[key]:
                report.missing_contexts.append(f"[{ctx}] {key}")

    report.unused_strings = sorted(set(strings_map.keys()) - extracted.strings)
    return report
