"""Unit tests for metadata tooltip feature."""
import pytest
from PySide6.QtCore import Qt, QPoint, QEvent
from PySide6.QtGui import QMouseEvent, QKeyEvent
from PySide6.QtWidgets import QApplication, QListWidgetItem, QToolTip

from spectroview.viewmodel.utils import (
    extract_primary_metadata,
    format_metadata_tooltip,
    MetadataTooltipManager,
)
from spectroview.view.components.v_spectra_list import VSpectraList
from spectroview.view.components.v_map_list import VMapsList


class TestMetadataExtraction:
    """Tests for extract_primary_metadata and format_metadata_tooltip."""

    def test_full_metadata(self):
        meta = {
            "Laser Wavelength (nm)": "532.00",
            "Laser Power (%)": "10.0",
            "Exposure Time (s)": "2.5",
            "Accumulations": 3,
            "Objective Used": "100x",
        }
        res = extract_primary_metadata(meta)
        assert res["Laser Wavelength"] == "532.00 nm"
        assert res["Laser Power"] == "10.0 %"
        assert res["Exposure Time"] == "2.5 s"
        assert res["Accumulation"] == "3"
        assert res["Objective"] == "100x"

    def test_units_already_present(self):
        meta = {
            "Laser Wavelength": "785 nm",
            "Laser Power": "50 mW",
            "Exposure Time": "10 s",
            "Accumulation": "1",
            "Objective": "50×",
        }
        res = extract_primary_metadata(meta)
        assert res["Laser Wavelength"] == "785 nm"
        assert res["Laser Power"] == "50 mW"
        assert res["Exposure Time"] == "10 s"
        assert res["Accumulation"] == "1"
        assert res["Objective"] == "50×"

    def test_spc_keys_fallback(self):
        meta = {
            "EXCIT_LINE": "632.8",
            "ND FILTER": "100",
            "ACQ. TIME (S)": "5",
            "ACCUMULATIONS": "4",
            "OBJECTIV": "x20",
        }
        res = extract_primary_metadata(meta)
        assert res["Laser Wavelength"] == "632.8 nm"
        assert res["Laser Power"] == "100"
        assert res["Exposure Time"] == "5 s"
        assert res["Accumulation"] == "4"
        assert res["Objective"] == "x20"

    def test_empty_or_unknown_metadata_graceful(self):
        meta = {
            "Laser Wavelength (nm)": "Unknown",
            "Laser Power (%)": "",
            "Exposure Time (s)": "none",
            "Accumulations": None,
            "Objective Used": "N/A",
        }
        res = extract_primary_metadata(meta)
        assert res["Laser Wavelength"] == "N/A"
        assert res["Laser Power"] == "N/A"
        assert res["Exposure Time"] == "N/A"
        assert res["Accumulation"] == "N/A"
        assert res["Objective"] == "N/A"

    def test_none_or_empty_dict(self):
        res = extract_primary_metadata(None)
        assert all(v == "N/A" for v in res.values())

        res2 = extract_primary_metadata({})
        assert all(v == "N/A" for v in res2.values())

    def test_object_with_metadata_attribute(self):
        class DummyItem:
            metadata = {
                "Laser Wavelength (nm)": "532",
                "Laser Power (%)": "25",
                "Exposure Time (s)": "1",
                "Accumulations": 2,
                "Objective Used": "50x",
            }
        res = extract_primary_metadata(DummyItem())
        assert res["Laser Wavelength"] == "532 nm"
        assert res["Laser Power"] == "25 %"
        assert res["Exposure Time"] == "1 s"
        assert res["Accumulation"] == "2"
        assert res["Objective"] == "50x"

    def test_format_metadata_tooltip(self):
        meta = {
            "Laser Wavelength (nm)": "532",
            "Laser Power (%)": "10",
            "Exposure Time (s)": "1",
            "Accumulations": 3,
            "Objective Used": "100×",
        }
        html = format_metadata_tooltip(meta)
        assert "<b>Metadata</b>" in html
        assert "Laser Wavelength: 532 nm" in html
        assert "Laser Power: 10 %" in html
        assert "Exposure Time: 1 s" in html
        assert "Accumulation: 3" in html
        assert "Objective: 100×" in html


class TestMetadataTooltipManager:
    """Tests for MetadataTooltipManager event filter interactions."""

    def test_hover_without_ctrl_shows_tooltip(self, qtbot, monkeypatch):
        """Hovering over an item starts delay timer, and displays tooltip upon timeout."""
        lw = VSpectraList()
        qtbot.addWidget(lw)
        lw.set_spectra_names([
            {
                "fname": "Spectrum_A",
                "is_active": True,
                "is_cropped": False,
                "has_baseline": False,
                "has_fit": False,
                "fit_success": False,
                "metadata": {
                    "Laser Wavelength (nm)": "532.0",
                    "Laser Power (%)": "5.0",
                    "Exposure Time (s)": "2.0",
                    "Accumulations": 1,
                    "Objective Used": "100x",
                },
            }
        ])
        lw.resize(300, 200)
        lw.show()

        monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.NoModifier))

        item = lw.item(0)
        rect = lw.visualItemRect(item)
        pos = rect.center()

        event = QMouseEvent(
            QEvent.MouseMove,
            pos,
            lw.viewport().mapToGlobal(pos),
            Qt.NoButton,
            Qt.NoButton,
            Qt.NoModifier,
        )
        QApplication.sendEvent(lw.viewport(), event)
        # Cold start: not immediately showing, timer active
        assert not lw._tooltip_manager._is_showing
        assert lw._tooltip_manager._delay_timer.isActive()
        assert lw._tooltip_manager._pending_item == item

        # Once delay timer fires, tooltip is displayed
        lw._tooltip_manager._delay_timer.timeout.emit()
        assert lw._tooltip_manager._is_showing
        assert lw._tooltip_manager._hovered_item == item

    def test_continuous_hover_to_other_spectra_is_immediate(self, qtbot, monkeypatch):
        """Once tooltip is displayed, moving to other items displays immediately with 0 delay."""
        lw = VSpectraList()
        qtbot.addWidget(lw)
        lw.set_spectra_names([
            {
                "fname": "Spectrum_A",
                "is_active": True,
                "is_cropped": False,
                "has_baseline": False,
                "has_fit": False,
                "fit_success": False,
                "metadata": {"Laser Wavelength (nm)": "532.0"},
            },
            {
                "fname": "Spectrum_B",
                "is_active": True,
                "is_cropped": False,
                "has_baseline": False,
                "has_fit": False,
                "fit_success": False,
                "metadata": {"Laser Wavelength (nm)": "785.0"},
            },
        ])
        lw.resize(300, 200)
        lw.show()

        monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.NoModifier))

        item_a = lw.item(0)
        item_b = lw.item(1)

        # 1. Hover item A, wait for delay timeout
        pos_a = lw.visualItemRect(item_a).center()
        event_a = QMouseEvent(QEvent.MouseMove, pos_a, lw.viewport().mapToGlobal(pos_a), Qt.NoButton, Qt.NoButton, Qt.NoModifier)
        QApplication.sendEvent(lw.viewport(), event_a)
        assert not lw._tooltip_manager._is_showing
        lw._tooltip_manager._delay_timer.timeout.emit()
        assert lw._tooltip_manager._is_showing
        assert lw._tooltip_manager._hovered_item == item_a

        # 2. Move directly to item B: warm state -> IMMEDIATELY displayed with NO delay
        pos_b = lw.visualItemRect(item_b).center()
        event_b = QMouseEvent(QEvent.MouseMove, pos_b, lw.viewport().mapToGlobal(pos_b), Qt.NoButton, Qt.NoButton, Qt.NoModifier)
        QApplication.sendEvent(lw.viewport(), event_b)
        assert lw._tooltip_manager._is_showing
        assert lw._tooltip_manager._hovered_item == item_b
        assert not lw._tooltip_manager._delay_timer.isActive()

    def test_leave_before_delay_cancels_tooltip(self, qtbot, monkeypatch):
        """Leaving the widget before delay expires cancels the tooltip."""
        lw = VSpectraList()
        qtbot.addWidget(lw)
        lw.set_spectra_names([
            {"fname": "Spectrum_A", "is_active": True, "is_cropped": False, "has_baseline": False, "has_fit": False, "fit_success": False, "metadata": {"Laser Wavelength (nm)": "532.0"}}
        ])
        lw.resize(300, 200)
        lw.show()

        monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.NoModifier))
        item = lw.item(0)
        pos = lw.visualItemRect(item).center()

        event = QMouseEvent(QEvent.MouseMove, pos, lw.viewport().mapToGlobal(pos), Qt.NoButton, Qt.NoButton, Qt.NoModifier)
        QApplication.sendEvent(lw.viewport(), event)
        assert lw._tooltip_manager._delay_timer.isActive()

        # Leave before timeout
        leave_event = QEvent(QEvent.Leave)
        QApplication.sendEvent(lw.viewport(), leave_event)

        assert not lw._tooltip_manager._delay_timer.isActive()
        assert lw._tooltip_manager._pending_item is None
        assert not lw._tooltip_manager._is_showing

    def test_require_ctrl_mode_behavior(self, qtbot, monkeypatch):
        """When require_ctrl=True, tooltip requires Ctrl to show and hides on Ctrl release."""
        lw = VSpectraList()
        qtbot.addWidget(lw)
        lw._tooltip_manager.require_ctrl = True
        lw.set_spectra_names([
            {
                "fname": "Spectrum_A",
                "is_active": True,
                "is_cropped": False,
                "has_baseline": False,
                "has_fit": False,
                "fit_success": False,
                "metadata": {"Laser Wavelength (nm)": "532.0"},
            }
        ])
        lw.resize(300, 200)
        lw.show()

        item = lw.item(0)
        rect = lw.visualItemRect(item)
        pos = rect.center()

        # 1. Hover without Ctrl -> no tooltip
        monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.NoModifier))
        event = QMouseEvent(
            QEvent.MouseMove,
            pos,
            lw.viewport().mapToGlobal(pos),
            Qt.NoButton,
            Qt.NoButton,
            Qt.NoModifier,
        )
        QApplication.sendEvent(lw.viewport(), event)
        assert not lw._tooltip_manager._is_showing

        # 2. Hover with Ctrl -> shows tooltip after delay
        monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.ControlModifier))
        event_ctrl = QMouseEvent(
            QEvent.MouseMove,
            pos,
            lw.viewport().mapToGlobal(pos),
            Qt.NoButton,
            Qt.NoButton,
            Qt.ControlModifier,
        )
        QApplication.sendEvent(lw.viewport(), event_ctrl)
        if lw._tooltip_manager._delay_timer.isActive():
            lw._tooltip_manager._delay_timer.timeout.emit()
        assert lw._tooltip_manager._is_showing

        # 3. Release Ctrl -> hides tooltip
        monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.NoModifier))
        key_event = QKeyEvent(QEvent.KeyRelease, Qt.Key_Control, Qt.NoModifier)
        QApplication.sendEvent(lw, key_event)
        assert not lw._tooltip_manager._is_showing

    def test_tooltip_persists_when_mouse_stationary(self, qtbot, monkeypatch):
        """Stationary mouse fires QHelpEvent (QEvent.ToolTip) which must not dismiss the tooltip."""
        from PySide6.QtGui import QHelpEvent
        lw = VSpectraList()
        qtbot.addWidget(lw)
        lw.set_spectra_names([
            {"fname": "Spectrum_A", "is_active": True, "is_cropped": False, "has_baseline": False, "has_fit": False, "fit_success": False, "metadata": {"Laser Wavelength (nm)": "532"}}
        ])
        lw.resize(300, 200)
        lw.show()

        item = lw.item(0)
        rect = lw.visualItemRect(item)
        pos = rect.center()
        gpos = lw.viewport().mapToGlobal(pos)

        event = QMouseEvent(
            QEvent.MouseMove,
            pos,
            gpos,
            Qt.NoButton,
            Qt.NoButton,
            Qt.NoModifier,
        )
        QApplication.sendEvent(lw.viewport(), event)
        if lw._tooltip_manager._delay_timer.isActive():
            lw._tooltip_manager._delay_timer.timeout.emit()
        assert lw._tooltip_manager._is_showing

        # Stationary mouse trigger: QHelpEvent
        help_event = QHelpEvent(QEvent.ToolTip, pos, gpos)
        QApplication.sendEvent(lw.viewport(), help_event)

        # Tooltip MUST remain showing!
        assert lw._tooltip_manager._is_showing
        assert lw._tooltip_manager._popup.isVisible()

    def test_leave_event_hides_tooltip(self, qtbot, monkeypatch):
        lw = VSpectraList()
        qtbot.addWidget(lw)
        lw.set_spectra_names([
            {"fname": "Spectrum_A", "is_active": True, "is_cropped": False, "has_baseline": False, "has_fit": False, "fit_success": False, "metadata": {"Laser Wavelength (nm)": "532"}}
        ])
        lw.resize(300, 200)
        lw.show()

        monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.ControlModifier))
        item = lw.item(0)
        rect = lw.visualItemRect(item)
        pos = rect.center()

        event = QMouseEvent(
            QEvent.MouseMove,
            pos,
            lw.viewport().mapToGlobal(pos),
            Qt.NoButton,
            Qt.NoButton,
            Qt.ControlModifier,
        )
        QApplication.sendEvent(lw.viewport(), event)
        if lw._tooltip_manager._delay_timer.isActive():
            lw._tooltip_manager._delay_timer.timeout.emit()
        assert lw._tooltip_manager._is_showing

        leave_event = QEvent(QEvent.Leave)
        QApplication.sendEvent(lw.viewport(), leave_event)
        assert not lw._tooltip_manager._is_showing

    def test_external_hide_event_does_not_dismiss_tooltip(self, qtbot, monkeypatch):
        """Ensure animation or unassociated widget hide events do not dismiss tooltip."""
        from PySide6.QtWidgets import QLabel
        lw = VSpectraList()
        qtbot.addWidget(lw)
        lw.set_spectra_names([
            {"fname": "Spectrum_A", "is_active": True, "is_cropped": False, "has_baseline": False, "has_fit": False, "fit_success": False, "metadata": {"Laser Wavelength (nm)": "532"}}
        ])
        lw.resize(300, 200)
        lw.show()

        monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.ControlModifier))
        item = lw.item(0)
        rect = lw.visualItemRect(item)
        pos = rect.center()

        event = QMouseEvent(
            QEvent.MouseMove,
            pos,
            lw.viewport().mapToGlobal(pos),
            Qt.NoButton,
            Qt.NoButton,
            Qt.ControlModifier,
        )
        QApplication.sendEvent(lw.viewport(), event)
        if lw._tooltip_manager._delay_timer.isActive():
            lw._tooltip_manager._delay_timer.timeout.emit()
        assert lw._tooltip_manager._is_showing

        # Send Hide event on an unrelated widget (e.g. tooltip animation effect or popup)
        dummy_label = QLabel()
        qtbot.addWidget(dummy_label)
        hide_event = QEvent(QEvent.Hide)
        QApplication.sendEvent(dummy_label, hide_event)

        # Tooltip must remain showing
        assert lw._tooltip_manager._is_showing

    def test_press_ctrl_while_hovering_shows_tooltip(self, qtbot, monkeypatch):
        lw = VSpectraList()
        qtbot.addWidget(lw)
        lw.set_spectra_names([
            {"fname": "Spectrum_A", "is_active": True, "is_cropped": False, "has_baseline": False, "has_fit": False, "fit_success": False, "metadata": {"Laser Wavelength (nm)": "532"}}
        ])
        lw.resize(300, 200)
        lw.show()

        item = lw.item(0)
        rect = lw.visualItemRect(item)
        pos = rect.center()
        global_pos = lw.viewport().mapToGlobal(pos)

        # Mock cursor position to be over item 0
        from PySide6.QtGui import QCursor
        monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: global_pos))
        monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.ControlModifier))

        key_event = QKeyEvent(QEvent.KeyPress, Qt.Key_Control, Qt.ControlModifier)
        QApplication.sendEvent(lw, key_event)
        if lw._tooltip_manager._delay_timer.isActive():
            lw._tooltip_manager._delay_timer.timeout.emit()

        assert lw._tooltip_manager._is_showing
        assert lw._tooltip_manager._hovered_item == item

    def test_placeholder_item_does_not_show_tooltip(self, qtbot, monkeypatch):
        lw = VSpectraList()
        qtbot.addWidget(lw)
        lw.resize(300, 200)
        lw.show()

        monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.ControlModifier))
        item = lw.item(0)
        assert item.flags() == Qt.NoItemFlags

        rect = lw.visualItemRect(item)
        pos = rect.center()

        event = QMouseEvent(
            QEvent.MouseMove,
            pos,
            lw.viewport().mapToGlobal(pos),
            Qt.NoButton,
            Qt.NoButton,
            Qt.ControlModifier,
        )
        QApplication.sendEvent(lw.viewport(), event)
        assert not lw._tooltip_manager._is_showing


class TestMapsListMetadataTooltip:
    """Tests for metadata tooltip in VMapsList."""

    def test_maps_list_metadata_provider(self, qtbot, monkeypatch):
        maps_widget = VMapsList()
        qtbot.addWidget(maps_widget)
        maps_widget.set_maps_names(["Map_1", "Map_2"])
        maps_widget.resize(400, 300)
        maps_widget.show()

        mock_metadata = {
            "Map_1": {
                "Laser Wavelength (nm)": "532.0",
                "Laser Power (%)": "100",
                "Exposure Time (s)": "0.5",
                "Accumulations": 2,
                "Objective Used": "50x",
            }
        }
        maps_widget.set_map_metadata_provider(lambda name: mock_metadata.get(name, {}))

        monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.ControlModifier))

        item = maps_widget.maps_list.item(0)
        assert item.text() == "Map_1"
        rect = maps_widget.maps_list.visualItemRect(item)
        pos = rect.center()

        event = QMouseEvent(
            QEvent.MouseMove,
            pos,
            maps_widget.maps_list.viewport().mapToGlobal(pos),
            Qt.NoButton,
            Qt.NoButton,
            Qt.ControlModifier,
        )
        QApplication.sendEvent(maps_widget.maps_list.viewport(), event)
        if maps_widget._maps_tooltip_manager._delay_timer.isActive():
            maps_widget._maps_tooltip_manager._delay_timer.timeout.emit()

        assert maps_widget._maps_tooltip_manager._is_showing
        assert maps_widget._maps_tooltip_manager._hovered_item == item
