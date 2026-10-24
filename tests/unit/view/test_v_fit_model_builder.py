"""Unit tests for VFitModelBuilder drag and drop."""
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
        assert dropped_paths == [str(json_file)]

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
