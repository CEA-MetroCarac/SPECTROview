"""Centralized translation manager and QTranslator lifecycle for SPECTROview."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QObject, QTranslator, Signal

from spectroview import resource_path

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("en", "English"),
    ("fr", "Français"),
)
DEFAULT_LANGUAGE: str = "en"

_VALID_LANG_CODES = {code for code, _ in SUPPORTED_LANGUAGES}


def normalize_language(lang: Optional[str]) -> str:
    """Normalize arbitrary locale/language strings to canonical codes (``'en'``, ``'fr'``)."""
    raw = str(lang or "").strip().lower().replace("-", "_")
    if not raw:
        return DEFAULT_LANGUAGE
    if raw in _VALID_LANG_CODES:
        return raw
    if raw.startswith("fr") or raw in ("french", "français", "francais"):
        return "fr"
    if raw.startswith("en") or raw in ("english", "anglais"):
        return "en"
    return DEFAULT_LANGUAGE


def _format_safe(template: str, fallback: str, kwargs: dict[str, Any]) -> str:
    """Format a translated template safely, falling back on placeholder mismatch."""
    if not kwargs:
        return template
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError, IndexError, TypeError):
        try:
            return fallback.format(**kwargs)
        except Exception:
            return fallback


def _plural_index(lang: str, count: int) -> int:
    """Return plural form index (0 = singular, 1 = plural) for *lang*."""
    n = abs(count)
    if lang == "fr":
        return 0 if n in (0, 1) else 1
    return 0 if n == 1 else 1


class TranslationManager(QObject):
    """Singleton manager owning the active GUI language, JSON catalogs, and Qt QTranslator."""

    language_changed = Signal(str)

    _instance: Optional[TranslationManager] = None

    @classmethod
    def instance(cls) -> TranslationManager:
        """Return the active singleton instance, creating a default one if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, settings: Any = None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        prev = TranslationManager._instance
        if prev is not None and prev is not self:
            prev._uninstall_qt_translator()
        TranslationManager._instance = self

        self._settings = settings
        self._current_lang: str = DEFAULT_LANGUAGE
        self._catalog: dict[str, Any] = {}
        self._catalog_cache: dict[str, dict[str, Any]] = {}
        self._qt_translator: Optional[QTranslator] = None

    def bind_settings(self, settings: Any) -> None:
        """Attach an ``MSettings`` instance used when persisting language changes."""
        self._settings = settings

    @property
    def current_language(self) -> str:
        """Return the active canonical language code (e.g. ``'en'`` or ``'fr'``)."""
        return self._current_lang

    @property
    def catalog(self) -> dict[str, Any]:
        """Return the currently active translation catalog dictionary."""
        return self._catalog

    def _i18n_dir(self) -> Path:
        return Path(resource_path("spectroview/resources/i18n"))

    def _load_catalog(self, lang: str) -> dict[str, Any]:
        if lang == DEFAULT_LANGUAGE:
            return {}
        if lang in self._catalog_cache:
            return self._catalog_cache[lang]

        catalog_path = self._i18n_dir() / f"{lang}.json"
        if not catalog_path.is_file():
            logger.warning("Translation catalog not found: %s", catalog_path)
            return {}
        try:
            data = json.loads(catalog_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._catalog_cache[lang] = data
                return data
        except Exception as exc:
            logger.warning("Failed to load translation catalog %s: %s", catalog_path, exc)
        return {}

    def _uninstall_qt_translator(self) -> None:
        app = QCoreApplication.instance()
        if self._qt_translator is not None:
            if app is not None:
                try:
                    app.removeTranslator(self._qt_translator)
                except RuntimeError:
                    pass
            self._qt_translator = None

    def _install_qt_translator(self, lang: str) -> None:
        self._uninstall_qt_translator()
        if lang == DEFAULT_LANGUAGE:
            return
        app = QCoreApplication.instance()
        if app is None:
            return

        translator = QTranslator(self)
        qm_name = f"qtbase_{lang}"
        qt_dir = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        loaded = translator.load(qm_name, qt_dir)
        if not loaded:
            loaded = translator.load(qm_name, str(self._i18n_dir()))
        if loaded:
            app.installTranslator(translator)
            self._qt_translator = translator
        else:
            logger.debug("Qt standard translator '%s' not found in %s", qm_name, qt_dir)

    def translate(self, text: str, *, ctx: str = "", **kwargs: Any) -> str:
        """Translate *text* into the active language, optionally using *ctx* and format *kwargs*."""
        if not text:
            return ""
        if self._current_lang == DEFAULT_LANGUAGE:
            return _format_safe(text, text, kwargs)

        template = ""
        if ctx:
            ctx_map = self._catalog.get("contexts", {}).get(ctx, {})
            if isinstance(ctx_map, dict):
                template = ctx_map.get(text, "")
        if not template:
            strings_map = self._catalog.get("strings", {})
            if isinstance(strings_map, dict):
                template = strings_map.get(text, "")
        if not template:
            template = text
        return _format_safe(template, text, kwargs)

    def translate_plural(
        self,
        singular: str,
        plural: str,
        count: int,
        *,
        ctx: str = "",
        **kwargs: Any,
    ) -> str:
        """Translate a countable string using locale-appropriate pluralization rules."""
        fmt_kwargs = {"count": count, **kwargs}
        idx = _plural_index(self._current_lang, count)
        default_template = singular if idx == 0 else plural

        if self._current_lang == DEFAULT_LANGUAGE:
            return _format_safe(default_template, default_template, fmt_kwargs)

        key = f"{singular}|{plural}"
        forms = None
        if ctx:
            ctx_map = self._catalog.get("contexts", {}).get(ctx, {})
            if isinstance(ctx_map, dict):
                forms = ctx_map.get(key)
        if forms is None:
            plurals_map = self._catalog.get("plurals", {})
            if isinstance(plurals_map, dict):
                forms = plurals_map.get(key)

        template = ""
        if isinstance(forms, (list, tuple)) and len(forms) > idx and forms[idx]:
            template = str(forms[idx])
        if not template:
            template = default_template
        return _format_safe(template, default_template, fmt_kwargs)

    def set_language(self, lang: str, *, persist: bool = True) -> str:
        """Switch the active language, update Qt translator, refresh bindings, and emit signal."""
        norm = normalize_language(lang)
        self._current_lang = norm
        self._catalog = self._load_catalog(norm)
        self._install_qt_translator(norm)

        if persist and self._settings is not None:
            self._settings.set_language(norm)

        from spectroview.i18n.bindings import retranslate_all

        retranslate_all()
        self.language_changed.emit(norm)
        return norm


def get_language() -> str:
    """Return the active GUI language code (e.g. ``'en'`` or ``'fr'``)."""
    return TranslationManager.instance().current_language


def set_language(lang: str, *, persist: bool = True) -> str:
    """Switch the active GUI language and retranslate all live bound widgets."""
    return TranslationManager.instance().set_language(lang, persist=persist)


def tr(text: str, *, ctx: str = "", **kwargs: Any) -> str:
    """Translate *text* into the currently active GUI language."""
    return TranslationManager.instance().translate(text, ctx=ctx, **kwargs)


def tr_plural(
    singular: str,
    plural: str,
    count: int,
    *,
    ctx: str = "",
    **kwargs: Any,
) -> str:
    """Translate a singular/plural pair according to *count* and active language rules."""
    return TranslationManager.instance().translate_plural(
        singular, plural, count, ctx=ctx, **kwargs
    )
