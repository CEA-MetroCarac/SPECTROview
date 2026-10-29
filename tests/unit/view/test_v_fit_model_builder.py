"""Unit tests for VFitModelBuilder drag and drop."""
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QMimeData, QPoint
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent

from spectroview.view.components.v_fit_model_builder import VFitModelBuilder


class TestVFitModelBuilderDragDrop:
    def test_accept_drops_and_tooltips(self, qapp):
        builder = VFitModelBuilder()
        assert builder.acceptDrops() is True
        assert "drag and drop" in builder.cbb_model.toolTip().lower()
        assert "drag and drop" in builder.btn_load.toolTip().lower()

    def test_drag_enter_and_move_accepts_json(self, qapp, tmp_path):
        builder = VFitModelBuilder()
        json_file = tmp_path / "model.json"
        json_file.write_text("{}")

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(json_file))])

        enter_event = QDragEnterEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        builder.dragEnterEvent(enter_event)
        assert enter_event.isAccepted()

        move_event = QDragMoveEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        builder.dragMoveEvent(move_event)
        assert move_event.isAccepted()

    def test_drag_enter_ignores_non_json(self, qapp, tmp_path):
        builder = VFitModelBuilder()
        txt_file = tmp_path / "spectrum.txt"
        txt_file.write_text("1 2")

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(txt_file))])

        enter_event = QDragEnterEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        builder.dragEnterEvent(enter_event)
        assert not enter_event.isAccepted()

    def test_drop_event_emits_signal(self, qapp, tmp_path):
        builder = VFitModelBuilder()
        json_file = tmp_path / "my_model.json"
        json_file.write_text("{}")

        dropped_paths = []
        builder.fit_model_dropped.connect(dropped_paths.append)

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(json_file))])

        drop_event = QDropEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        builder.dropEvent(drop_event)

        assert drop_event.isAccepted()
        assert [Path(p) for p in dropped_paths] == [json_file]

    def test_drop_event_ignores_non_json(self, qapp, tmp_path):
        builder = VFitModelBuilder()
        txt_file = tmp_path / "spectrum.txt"
        txt_file.write_text("1 2")

        dropped_paths = []
        builder.fit_model_dropped.connect(dropped_paths.append)

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(txt_file))])

        drop_event = QDropEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        builder.dropEvent(drop_event)

        assert not drop_event.isAccepted()
        assert dropped_paths == []


class TestVFitModelBuilderCopyPasteFeedback:
    def test_copy_paste_visual_feedback(self, qapp):
        builder = VFitModelBuilder()

        # Baseline copy/paste
        base_copy_emitted = []
        builder.baseline_copy_requested.connect(lambda: base_copy_emitted.append(True))
        builder.btn_base_copy.click()
        assert len(base_copy_emitted) == 1

        base_paste_emitted = []
        builder.baseline_paste_requested.connect(base_paste_emitted.append)
        builder.btn_base_paste.click()
        assert len(base_paste_emitted) == 1

        # Peak copy/paste
        peak_copy_emitted = []
        builder.peaks_copy_requested.connect(lambda: peak_copy_emitted.append(True))
        builder.btn_peak_copy.click()
        assert len(peak_copy_emitted) == 1

        peak_paste_emitted = []
        builder.peaks_paste_requested.connect(peak_paste_emitted.append)
        builder.btn_peak_paste.click()
        assert len(peak_paste_emitted) == 1

        # Fit button size (increased by 50% from 54px to 81px)
        assert builder.btn_fit.width() == 81
        assert builder.btn_fit.height() == 24

        # Apply fit model button size (increased by 30% from 66px to 86px)
        assert builder.btn_apply.width() == 86

        # Fit model combobox size (reduced maximum width by 10% from 336px to 302px)
        assert builder.cbb_model.maximumWidth() == 302
        assert builder.btn_refresh.width() == 30

        # Fit model copy/paste (icon feedback without changing button text)
        fit_copy_emitted = []
        builder.fitmodel_copy_requested.connect(lambda: fit_copy_emitted.append(True))
        builder.btn_copy.click()
        assert len(fit_copy_emitted) == 1
        assert builder.btn_copy.text() == "Copy"
        assert not builder.btn_copy.icon().isNull()

        fit_paste_emitted = []
        builder.fitmodel_paste_requested.connect(fit_paste_emitted.append)
        builder.btn_paste.click()
        assert len(fit_paste_emitted) == 1
        assert builder.btn_paste.text() == "Paste"
        assert not builder.btn_paste.icon().isNull()
