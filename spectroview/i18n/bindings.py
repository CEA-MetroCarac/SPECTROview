"""Weak-referenced declarative UI bindings for leak-free live language switching."""

from __future__ import annotations

from typing import Any, Callable, Sequence
import weakref

from PySide6.QtCore import QObject
from shiboken6 import isValid

from spectroview.i18n.manager import tr, tr_plural

# Registry maps (id(obj), slot_name) -> (weakref_to_obj, callable(obj))
_BINDINGS: dict[tuple[int, str], tuple[weakref.ref[Any], Callable[[Any], None]]] = {}
_CALLBACKS: dict[tuple[int, str], tuple[weakref.ref[Any], Callable[[], None]]] = {}


def _is_alive(obj: Any) -> bool:
    if obj is None:
        return False
    if isinstance(obj, QObject):
        return isValid(obj)
    return True


def _register_binding(
    obj: Any, slot: str, updater: Callable[[Any], None]
) -> None:
    """Apply *updater* immediately and register it under ``(id(obj), slot)``."""
    updater(obj)
    key = (id(obj), slot)
    try:
        ref = weakref.ref(obj)
    except TypeError:
        return
    _BINDINGS[key] = (ref, updater)


def bind_text(obj: Any, source: str, *, ctx: str = "", **kwargs: Any) -> Any:
    """Bind ``obj.setText(tr(source, ...))`` for immediate and live updates."""
    _register_binding(
        obj,
        "text",
        lambda w: w.setText(tr(source, ctx=ctx, **kwargs)),
    )
    return obj


def bind_plural_text(
    obj: Any,
    singular: str,
    plural: str,
    count: int,
    *,
    ctx: str = "",
    **kwargs: Any,
) -> Any:
    """Bind ``obj.setText(tr_plural(...))`` and overwrite any prior ``text`` binding."""
    _register_binding(
        obj,
        "text",
        lambda w: w.setText(tr_plural(singular, plural, count, ctx=ctx, **kwargs)),
    )
    return obj


def bind_tooltip(obj: Any, source: str, *, ctx: str = "", **kwargs: Any) -> Any:
    """Bind ``obj.setToolTip(tr(source, ...))`` for immediate and live updates."""
    _register_binding(
        obj,
        "tooltip",
        lambda w: w.setToolTip(tr(source, ctx=ctx, **kwargs)),
    )
    return obj


def bind_placeholder(obj: Any, source: str, *, ctx: str = "", **kwargs: Any) -> Any:
    """Bind ``obj.setPlaceholderText(tr(source, ...))`` for immediate and live updates."""
    _register_binding(
        obj,
        "placeholder",
        lambda w: w.setPlaceholderText(tr(source, ctx=ctx, **kwargs)),
    )
    return obj


def bind_title(obj: Any, source: str, *, ctx: str = "", **kwargs: Any) -> Any:
    """Bind ``obj.setTitle(tr(source, ...))`` (e.g. ``QGroupBox``, ``QMenu``)."""
    _register_binding(
        obj,
        "title",
        lambda w: w.setTitle(tr(source, ctx=ctx, **kwargs)),
    )
    return obj


def bind_window_title(obj: Any, source: str, *, ctx: str = "", **kwargs: Any) -> Any:
    """Bind ``obj.setWindowTitle(tr(source, ...))`` for windows and dialogs."""
    _register_binding(
        obj,
        "window_title",
        lambda w: w.setWindowTitle(tr(source, ctx=ctx, **kwargs)),
    )
    return obj


def bind_tab_text(
    tab_widget: Any,
    page_widget: Any,
    source: str,
    *,
    ctx: str = "",
    **kwargs: Any,
) -> Any:
    """Bind a ``QTabWidget`` tab label keyed by child widget identity."""
    page_ref = weakref.ref(page_widget)

    def _apply(tw: Any) -> None:
        page = page_ref()
        if not _is_alive(page):
            return
        idx = tw.indexOf(page)
        if idx >= 0:
            tw.setTabText(idx, tr(source, ctx=ctx, **kwargs))

    _register_binding(tab_widget, f"tab:{id(page_widget)}", _apply)
    return page_widget


def bind_combo_items(
    combo: Any,
    items: Sequence[tuple[str, Any] | str],
    *,
    ctx: str = "",
    populate: bool = True,
) -> Any:
    """Bind ``QComboBox`` item labels while keeping canonical ``itemData`` untouched."""
    normalized: list[tuple[str, Any]] = [
        (item, item) if isinstance(item, str) else (item[0], item[1])
        for item in items
    ]

    if populate:
        was_blocked = combo.blockSignals(True)
        try:
            combo.clear()
            for label, data in normalized:
                combo.addItem(tr(label, ctx=ctx), data)
        finally:
            combo.blockSignals(was_blocked)

    def _apply(cb: Any) -> None:
        was_blocked = cb.blockSignals(True)
        try:
            for idx, (label, data) in enumerate(normalized):
                if idx >= cb.count():
                    break
                if cb.itemData(idx) == data or (data is None and idx == 0):
                    cb.setItemText(idx, tr(label, ctx=ctx))
        finally:
            cb.blockSignals(was_blocked)

    _register_binding(combo, "combo_items", _apply)
    return combo


def bind_header_labels(
    view: Any, sources: Sequence[str], *, ctx: str = ""
) -> Any:
    """Bind ``QTableWidget`` / ``QTreeWidget`` header labels."""
    src_tuple = tuple(sources)

    def _apply(v: Any) -> None:
        labels = [tr(s, ctx=ctx) for s in src_tuple]
        if hasattr(v, "setHorizontalHeaderLabels"):
            v.setHorizontalHeaderLabels(labels)
        elif hasattr(v, "setHeaderLabels"):
            v.setHeaderLabels(labels)

    _register_binding(view, "header_labels", _apply)
    return view


def bind_form_row(
    form_layout: Any,
    label_source: str,
    field: Any,
    *,
    ctx: str = "",
    tooltip: str = "",
) -> Any:
    """Create a bound ``QLabel`` row in a ``QFormLayout`` and return the label."""
    from PySide6.QtWidgets import QLabel

    parent = form_layout.parentWidget()
    lbl = QLabel(parent)
    bind_text(lbl, label_source, ctx=ctx)
    if tooltip:
        bind_tooltip(lbl, tooltip, ctx=ctx)
        bind_tooltip(field, tooltip, ctx=ctx)
    form_layout.addRow(lbl, field)
    return lbl


def on_language_changed(
    owner: Any,
    callback: Callable[[], None],
    *,
    slot_key: str = "retranslate",
) -> None:
    """Register a weak-referenced callback invoked on every language switch."""
    key = (id(owner), slot_key)
    try:
        ref = weakref.ref(owner)
    except TypeError:
        return
    _CALLBACKS[key] = (ref, callback)


def clear_dead_bindings() -> int:
    """Prune garbage-collected or C++-deleted widget entries; return number removed."""
    removed = 0
    for key in list(_BINDINGS):
        ref, _ = _BINDINGS[key]
        obj = ref()
        if not _is_alive(obj):
            del _BINDINGS[key]
            removed += 1
    for key in list(_CALLBACKS):
        ref, _ = _CALLBACKS[key]
        obj = ref()
        if not _is_alive(obj):
            del _CALLBACKS[key]
            removed += 1
    return removed


def retranslate_all() -> None:
    """Re-apply all live widget bindings and invoke live callbacks after language change."""
    for key in list(_BINDINGS):
        entry = _BINDINGS.get(key)
        if entry is None:
            continue
        ref, updater = entry
        obj = ref()
        if not _is_alive(obj):
            _BINDINGS.pop(key, None)
            continue
        try:
            updater(obj)
        except RuntimeError:
            _BINDINGS.pop(key, None)

    for key in list(_CALLBACKS):
        entry = _CALLBACKS.get(key)
        if entry is None:
            continue
        ref, cb = entry
        obj = ref()
        if not _is_alive(obj):
            _CALLBACKS.pop(key, None)
            continue
        try:
            cb()
        except RuntimeError:
            _CALLBACKS.pop(key, None)


def reset_for_tests() -> None:
    """Clear all registered UI bindings and callbacks (used in test teardown)."""
    _BINDINGS.clear()
    _CALLBACKS.clear()


def combo_data_or_text(combo: Any) -> Any:
    """Return canonical ``itemData`` if present, else ``currentText()``."""
    data = combo.currentData()
    return data if data is not None else combo.currentText()


def set_combo_by_data_or_text(combo: Any, value: Any) -> bool:
    """Select the combobox item matching *value* by ``itemData`` first, then text."""
    if value is None:
        return False
    idx = combo.findData(value)
    if idx < 0:
        idx = combo.findText(str(value))
    if idx >= 0:
        combo.setCurrentIndex(idx)
        return True
    return False
