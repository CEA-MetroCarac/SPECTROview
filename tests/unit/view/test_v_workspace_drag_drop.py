"""Unit and integration tests for drag-and-drop fit models in Spectra and Maps workspaces."""
import json

import pytest
from PySide6.QtCore import Qt, QUrl, QMimeData, QPoint
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from spectroview.view.v_workspace_spectra import VWorkspaceSpectra
from spectroview.view.v_workspace_maps import VWorkspaceMaps


@pytest.fixture
def sample_fit_model_file(tmp_path):
    model_path = tmp_path / "dragged_fit_model.json"
    content = {
        "0": {
            "fname": "spec1",
            "range_min": 450.0,
            "range_max": 550.0,
            "peak_labels": ["Si"],
            "fit_params": {"method": "Leastsq"},
            "baseline": {"mode": "Linear", "coef": 5},
            "peak_models": {
                "0": {
                    "Lorentzian": {
                        "ampli": {"value": 50.0, "min": 0, "max": 100},
                        "fwhm": {"value": 4.0, "min": 0, "max": 20},
                        "x0": {"value": 520.0, "min": 500, "max": 540},
                    }
                }
            },
        }
    }
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(content, f)
    return model_path


class TestSpectraWorkspaceDragDrop:
    def test_accept_drops_enabled(self, qapp):
        ws = VWorkspaceSpectra()
        assert ws.acceptDrops() is True

    def test_load_fit_model_file(self, qapp, sample_fit_model_file):
        ws = VWorkspaceSpectra()
        # Switch tab away first
        ws.bottom_tabs.setCurrentWidget(ws.v_fit_results)
        assert ws.bottom_tabs.currentWidget() == ws.v_fit_results

        success = ws.load_fit_model_file(str(sample_fit_model_file))
        assert success is True
        # Verify tab switched to fit model builder
        assert ws.bottom_tabs.currentWidget() == ws.v_fit_model_builder
        # Verify combobox updated to model
        assert ws.v_fit_model_builder.cbb_model.currentText() == "dragged_fit_model.json"

    def test_drag_and_drop_json_on_workspace(self, qapp, sample_fit_model_file):
        ws = VWorkspaceSpectra()
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(sample_fit_model_file))])

        enter_event = QDragEnterEvent(QPoint(50, 50), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        ws.dragEnterEvent(enter_event)
        assert enter_event.isAccepted()

        drop_event = QDropEvent(QPoint(50, 50), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        ws.dropEvent(drop_event)
        assert drop_event.isAccepted()

        assert ws.bottom_tabs.currentWidget() == ws.v_fit_model_builder
        assert ws.v_fit_model_builder.cbb_model.currentText() == "dragged_fit_model.json"

    def test_drag_non_json_is_ignored(self, qapp, tmp_path):
        ws = VWorkspaceSpectra()
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("1 2")

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(txt_file))])

        enter_event = QDragEnterEvent(QPoint(50, 50), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        ws.dragEnterEvent(enter_event)
        assert not enter_event.isAccepted()

        drop_event = QDropEvent(QPoint(50, 50), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        ws.dropEvent(drop_event)
        assert not drop_event.isAccepted()

    def test_fit_model_builder_signal_connected(self, qapp, sample_fit_model_file):
        ws = VWorkspaceSpectra()
        ws.bottom_tabs.setCurrentWidget(ws.v_fit_results)

        ws.v_fit_model_builder.fit_model_dropped.emit(str(sample_fit_model_file))
        assert ws.bottom_tabs.currentWidget() == ws.v_fit_model_builder
        assert ws.v_fit_model_builder.cbb_model.currentText() == "dragged_fit_model.json"


class TestMapsWorkspaceDragDrop:
    def test_accept_drops_enabled(self, qapp):
        ws = VWorkspaceMaps()
        assert ws.acceptDrops() is True

    def test_load_fit_model_file(self, qapp, sample_fit_model_file):
        ws = VWorkspaceMaps()
        ws.bottom_tabs.setCurrentWidget(ws.v_fit_results)
        assert ws.bottom_tabs.currentWidget() == ws.v_fit_results

        success = ws.load_fit_model_file(str(sample_fit_model_file))
        assert success is True
        assert ws.bottom_tabs.currentWidget() == ws.v_fit_model_builder
        assert ws.v_fit_model_builder.cbb_model.currentText() == "dragged_fit_model.json"

    def test_drag_and_drop_json_on_maps_workspace(self, qapp, sample_fit_model_file):
        ws = VWorkspaceMaps()
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(sample_fit_model_file))])

        enter_event = QDragEnterEvent(QPoint(50, 50), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        ws.dragEnterEvent(enter_event)
        assert enter_event.isAccepted()

        drop_event = QDropEvent(QPoint(50, 50), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        ws.dropEvent(drop_event)
        assert drop_event.isAccepted()

        assert ws.bottom_tabs.currentWidget() == ws.v_fit_model_builder
        assert ws.v_fit_model_builder.cbb_model.currentText() == "dragged_fit_model.json"

    def test_fit_model_builder_signal_connected_maps(self, qapp, sample_fit_model_file):
        ws = VWorkspaceMaps()
        ws.bottom_tabs.setCurrentWidget(ws.v_fit_results)

        ws.v_fit_model_builder.fit_model_dropped.emit(str(sample_fit_model_file))
        assert ws.bottom_tabs.currentWidget() == ws.v_fit_model_builder
        assert ws.v_fit_model_builder.cbb_model.currentText() == "dragged_fit_model.json"


class TestMainDragDropJson:
    def test_main_load_json_into_active_spectra_workspace(self, qapp, monkeypatch, sample_fit_model_file):
        from spectroview.main import Main
        monkeypatch.setattr(Main, "_prewarm_ai_chat", lambda self: None)
        win = Main()
        try:
            win.tabWidget.setCurrentWidget(win.v_spectra_workspace)
            win._load_files_by_paths([str(sample_fit_model_file)])

            assert win.tabWidget.currentWidget() == win.v_spectra_workspace
            assert win.v_spectra_workspace.bottom_tabs.currentWidget() == win.v_spectra_workspace.v_fit_model_builder
            assert win.v_spectra_workspace.v_fit_model_builder.cbb_model.currentText() == "dragged_fit_model.json"
        finally:
            win.close()

    def test_main_load_json_into_active_maps_workspace(self, qapp, monkeypatch, sample_fit_model_file):
        from spectroview.main import Main
        monkeypatch.setattr(Main, "_prewarm_ai_chat", lambda self: None)
        win = Main()
        try:
            win.tabWidget.setCurrentWidget(win.v_maps_workspace)
            win._load_files_by_paths([str(sample_fit_model_file)])

            assert win.tabWidget.currentWidget() == win.v_maps_workspace
            assert win.v_maps_workspace.bottom_tabs.currentWidget() == win.v_maps_workspace.v_fit_model_builder
            assert win.v_maps_workspace.v_fit_model_builder.cbb_model.currentText() == "dragged_fit_model.json"
        finally:
            win.close()

