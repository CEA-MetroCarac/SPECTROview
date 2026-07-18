"""Tests for view/components/v_export_dialog.py - VExportDialog.

Real, already-plotted VGraph (real dataset_Excel.xlsx data), headless,
mirroring test_customize_graph_dialog.py's pattern. QFileDialog.getSaveFileName
is monkeypatched (it would otherwise block forever under offscreen mode).
"""
import os
import tempfile

import pandas as pd
import pytest

from spectroview.view.components.v_graph import VGraph
from spectroview.view.components.v_export_dialog import VExportDialog, VBatchExportDialog


@pytest.fixture(scope="module")
def excel_df(dataframe_excel_file):
    if not dataframe_excel_file.exists():
        pytest.skip("dataset_Excel.xlsx not present")
    return pd.read_excel(dataframe_excel_file, sheet_name="sheet1")


def _plotted_graph(qapp, excel_df, plot_style="scatter", x="x0_Si", y=None, z=None):
    vg = VGraph(graph_id=1)
    vg.create_plot_widget(dpi=72)
    vg.df_name = "sheet1"
    vg.x = x
    vg.y = y if y is not None else ["ampli_Si"]
    vg.z = z
    vg.plot_style = plot_style
    vg.plot(excel_df)
    return vg


class TestVExportDialogDefaults:
    def test_defaults_to_graphs_own_figure_theme(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.figure_theme = "dark"
        dialog = VExportDialog(vg)
        assert dialog.panel.combo_theme.currentText() == "Dark Mode"

    def test_default_format_and_dpi_match_settings_defaults(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        assert dialog.panel.combo_format.currentText() == "PNG"
        assert dialog.panel.spin_dpi.value() == 300
        assert dialog.panel.cb_transparent.isChecked() is False


class TestVExportDialogExport:
    def test_export_png_creates_a_file(self, qapp, excel_df, monkeypatch, tmp_path):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        target = str(tmp_path / "graph1.png")
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getSaveFileName",
            lambda *a, **k: (target, "PNG Image (*.png)"),
        )

        dialog._on_export_clicked()

        assert os.path.exists(target)
        assert os.path.getsize(target) > 0

    @pytest.mark.parametrize("fmt_display,ext", [
        ("PNG", "png"), ("TIFF", "tiff"), ("SVG", "svg"), ("PDF", "pdf"), ("EPS", "eps"),
    ])
    def test_every_format_produces_a_valid_nonempty_file(self, qapp, excel_df, monkeypatch, tmp_path, fmt_display, ext):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        dialog.panel.combo_format.setCurrentText(fmt_display)
        target = str(tmp_path / f"graph1.{ext}")
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getSaveFileName",
            lambda *a, **k: (target, ""),
        )

        dialog._on_export_clicked()

        assert os.path.exists(target)
        assert os.path.getsize(target) > 0

    def test_missing_extension_is_appended(self, qapp, excel_df, monkeypatch, tmp_path):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        target_no_ext = str(tmp_path / "graph1")
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getSaveFileName",
            lambda *a, **k: (target_no_ext, ""),
        )

        dialog._on_export_clicked()

        assert os.path.exists(target_no_ext + ".png")

    def test_cancel_in_file_dialog_exports_nothing(self, qapp, excel_df, monkeypatch, tmp_path):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getSaveFileName",
            lambda *a, **k: ("", ""),
        )

        dialog._on_export_clicked()  # should return early, no crash

        # tmp_path always contains the isolated-QSettings "qsettings/" dir
        # (see conftest.py's autouse _isolate_qsettings fixture); assert no
        # export file was written, not that the directory is fully empty.
        exported_files = [p for p in tmp_path.iterdir() if p.name != "qsettings"]
        assert exported_files == []

    def test_export_restores_graphs_original_theme_after_override(self, qapp, excel_df, monkeypatch, tmp_path):
        vg = _plotted_graph(qapp, excel_df)
        assert vg.figure_theme == "light"
        dialog = VExportDialog(vg)
        dialog.panel.combo_theme.setCurrentText("Dark Mode")
        target = str(tmp_path / "graph1.png")
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getSaveFileName",
            lambda *a, **k: (target, ""),
        )

        dialog._on_export_clicked()

        assert vg.figure_theme == "light"  # restored, not left as "dark"

    def test_export_settings_persist_across_dialog_opens(self, qapp, excel_df, monkeypatch, tmp_path):
        vg = _plotted_graph(qapp, excel_df)
        dialog1 = VExportDialog(vg)
        dialog1.panel.combo_format.setCurrentText("SVG")
        dialog1.panel.spin_dpi.setValue(600)
        dialog1.panel.cb_transparent.setChecked(True)
        target = str(tmp_path / "graph1.svg")
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getSaveFileName",
            lambda *a, **k: (target, ""),
        )
        dialog1._on_export_clicked()

        dialog2 = VExportDialog(vg)
        assert dialog2.panel.combo_format.currentText() == "SVG"
        assert dialog2.panel.spin_dpi.value() == 600
        assert dialog2.panel.cb_transparent.isChecked() is True

    def test_last_directory_updated_after_export(self, qapp, excel_df, monkeypatch, tmp_path):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        target = str(tmp_path / "graph1.png")
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getSaveFileName",
            lambda *a, **k: (target, ""),
        )

        dialog._on_export_clicked()

        assert dialog.settings.get_last_directory() == str(tmp_path)


class TestVExportDialogPhysicalSizing:
    def test_blank_by_default(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        assert dialog.spin_width.value() == dialog._UNSET
        assert dialog.spin_height.value() == dialog._UNSET
        assert dialog._width_height_mm() == (None, None)

    def test_loads_graphs_own_persisted_size(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        vg.export_width_mm = 89.0
        vg.export_height_mm = 65.0
        dialog = VExportDialog(vg)
        assert dialog.spin_width.value() == 89.0
        assert dialog.spin_height.value() == 65.0

    def test_journal_preset_fills_width_height_dpi(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        dialog.combo_preset.setCurrentText("Nature - single column (89 mm)")
        assert dialog.spin_width.value() == 89.0
        assert dialog.panel.spin_dpi.value() == 300

    def test_unit_toggle_converts_displayed_value(self, qapp, excel_df):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        dialog.spin_width.setValue(25.4)  # 1 inch in mm
        dialog.combo_unit.setCurrentText("in")
        assert dialog.spin_width.value() == pytest.approx(1.0)
        dialog.combo_unit.setCurrentText("mm")
        assert dialog.spin_width.value() == pytest.approx(25.4)

    def test_export_uses_requested_physical_size(self, qapp, excel_df, monkeypatch, tmp_path):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        dialog.spin_width.setValue(100.0)
        dialog.spin_height.setValue(50.0)
        target = str(tmp_path / "graph1.png")
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getSaveFileName",
            lambda *a, **k: (target, ""),
        )

        dialog._on_export_clicked()

        assert os.path.exists(target)
        # 100mm/25.4 x 50mm/25.4 inches, at whatever dpi was used
        from PIL import Image
        with Image.open(target) as img:
            width_in = img.width / img.info.get('dpi', (300, 300))[0]
            height_in = img.height / img.info.get('dpi', (300, 300))[1]
        assert width_in == pytest.approx(100.0 / 25.4, rel=0.05)
        assert height_in == pytest.approx(50.0 / 25.4, rel=0.05)

    def test_export_persists_size_back_to_graph_and_emits_signal(self, qapp, excel_df, monkeypatch, tmp_path):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        dialog.spin_width.setValue(89.0)
        dialog.spin_height.setValue(65.0)
        target = str(tmp_path / "graph1.png")
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getSaveFileName",
            lambda *a, **k: (target, ""),
        )

        received = []
        vg.properties_changed.connect(lambda gid, props: received.append(props))
        dialog._on_export_clicked()

        assert vg.export_width_mm == 89.0
        assert vg.export_height_mm == 65.0
        assert received[-1]["export_width_mm"] == 89.0

    def test_leaving_blank_does_not_change_graphs_on_screen_size(self, qapp, excel_df, monkeypatch, tmp_path):
        vg = _plotted_graph(qapp, excel_df)
        dialog = VExportDialog(vg)
        target = str(tmp_path / "graph1.png")
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getSaveFileName",
            lambda *a, **k: (target, ""),
        )
        original_size = vg.figure.get_size_inches().copy()

        dialog._on_export_clicked()

        assert (vg.figure.get_size_inches() == original_size).all()
        assert vg.export_width_mm is None


class TestVBatchExportDialog:
    def _two_graphs(self, qapp, excel_df):
        vg1 = _plotted_graph(qapp, excel_df)
        vg1.graph_id = 1
        vg2 = _plotted_graph(qapp, excel_df, plot_style="line")
        vg2.graph_id = 2
        return {1: vg1, 2: vg2}

    def test_export_all_creates_one_file_per_graph(self, qapp, excel_df, monkeypatch, tmp_path):
        widgets = self._two_graphs(qapp, excel_df)
        dialog = VBatchExportDialog(widgets)
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getExistingDirectory",
            lambda *a, **k: str(tmp_path),
        )

        dialog._on_export_all_clicked()

        exported = [p for p in tmp_path.iterdir() if p.suffix == ".png"]
        assert len(exported) == 2

    def test_cancel_folder_picker_exports_nothing(self, qapp, excel_df, monkeypatch, tmp_path):
        widgets = self._two_graphs(qapp, excel_df)
        dialog = VBatchExportDialog(widgets)
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getExistingDirectory",
            lambda *a, **k: "",
        )

        dialog._on_export_all_clicked()

        exported = [p for p in tmp_path.iterdir() if p.name != "qsettings"]
        assert exported == []

    def test_each_graph_respects_its_own_persisted_physical_size(self, qapp, excel_df, monkeypatch, tmp_path):
        widgets = self._two_graphs(qapp, excel_df)
        widgets[1].export_width_mm = 100.0
        widgets[1].export_height_mm = 50.0
        dialog = VBatchExportDialog(widgets)
        monkeypatch.setattr(
            "spectroview.view.components.v_export_dialog.QFileDialog.getExistingDirectory",
            lambda *a, **k: str(tmp_path),
        )

        dialog._on_export_all_clicked()

        exported = sorted(p for p in tmp_path.iterdir() if p.suffix == ".png")
        assert len(exported) == 2
        from PIL import Image
        sizes_in = []
        for p in exported:
            with Image.open(p) as img:
                dpi = img.info.get('dpi', (300, 300))
                sizes_in.append((img.width / dpi[0], img.height / dpi[1]))
        # One of the two exported files must match graph 1's explicit 100x50mm size.
        assert any(
            w == pytest.approx(100.0 / 25.4, rel=0.05) and h == pytest.approx(50.0 / 25.4, rel=0.05)
            for w, h in sizes_in
        )

    def test_export_all_with_zero_graphs_shows_info_message_via_workspace(self, qapp, excel_df):
        """VBatchExportDialog itself assumes a non-empty dict (the caller,
        VWorkspaceGraphs._on_export_all_clicked, is responsible for the
        empty-workspace guard) -- documented here for cross-reference."""
        dialog = VBatchExportDialog({})
        assert len(dialog.graph_widgets) == 0
