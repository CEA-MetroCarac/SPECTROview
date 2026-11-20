"""Unit tests for virtualized VSpectraList, SpectraListModel, and SpectrumItemDelegate."""

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListView

from spectroview.view.components.lists.v_spectra_list import (
    SpectraListModel,
    SpectrumItemDelegate,
    VSpectraList,
)


class TestSpectrumItemDelegate:
    def test_delegate_defaults(self, qapp):
        """Delegate initializes with circle shape and 8px size by default."""
        delegate = SpectrumItemDelegate()
        assert delegate.shape == "circle"
        assert delegate.size == 8
        assert delegate.margin_right == 10

    def test_list_set_indicator_shape(self, qapp):
        """VSpectraList allows switching indicator shape between circle, square, rounded_square."""
        lst = VSpectraList()
        assert isinstance(lst.itemDelegate(), SpectrumItemDelegate)
        assert lst.itemDelegate().shape == "circle"

        lst.set_indicator_shape("square")
        assert lst.itemDelegate().shape == "square"

        lst.set_indicator_shape("rounded_square")
        assert lst.itemDelegate().shape == "rounded_square"

    def test_render_items_with_and_without_status(self, qapp):
        """Delegate cleanly paints items with various status colors and transparent items."""
        lst = VSpectraList()
        sample_data = [
            {"fname": "Spectrum_Converged", "is_active": True, "has_fit": True, "fit_success": True},
            {"fname": "Spectrum_Unconverged", "is_active": True, "has_fit": True, "fit_success": False},
            {"fname": "Spectrum_Baselined", "is_active": True, "has_baseline": True},
            {"fname": "Spectrum_Cropped", "is_active": True, "is_cropped": True},
            {"fname": "Spectrum_Original", "is_active": True},
        ]
        lst.set_spectra_names(sample_data)
        assert lst.count() == 5

        # Check that background colors are set in model data
        item_converged = lst.item(0)
        assert item_converged.background().color().alpha() > 0

        item_orig = lst.item(4)
        assert item_orig.background().color().alpha() == 0

        # Verify painter rendering succeeds
        pix = lst.grab()
        assert not pix.isNull()
        assert pix.width() > 0
        assert pix.height() > 0


class TestVirtualizedVSpectraList:
    def test_virtualized_list_view_and_model_architecture(self, qapp):
        """VSpectraList subclasses QListView backed by SpectraListModel with placeholder support."""
        lst = VSpectraList()
        assert isinstance(lst, QListView)
        assert isinstance(lst.model(), SpectraListModel)

        # Empty list exposes 6 spacer rows + 1 centered placeholder row, all non-selectable
        assert lst.count() == 7
        assert lst.model().has_placeholder
        assert lst.item(0).flags() == Qt.NoItemFlags
        assert lst.item(6).flags() == Qt.NoItemFlags
        assert lst.selected_model_indices() == []

        # Populating spectra hides placeholder and enables uniform item sizes
        lst.set_spectra_names([{"fname": "A", "is_active": True}, {"fname": "B", "is_active": False}])
        assert not lst.model().has_placeholder
        assert lst.uniformItemSizes()
        assert lst.count() == 2
        assert lst.item(0).text() == "A"
        assert lst.item(1).checkState() == Qt.Unchecked

        # Clearing restores placeholder cleanly
        lst.set_spectra_names([])
        assert lst.model().has_placeholder
        assert not lst.uniformItemSizes()
        assert lst.count() == 7

    def test_selection_preservation_and_default_first_item(self, qapp):
        """set_spectra_names selects row 0 by default and preserves multi-selection by fname."""
        lst = VSpectraList()
        emissions = []
        lst.selection_changed.connect(emissions.append)

        lst.set_spectra_names(
            [
                {"fname": "s1", "is_active": True},
                {"fname": "s2", "is_active": True},
                {"fname": "s3", "is_active": True},
            ]
        )
        assert lst.selected_model_indices() == [0]
        assert emissions[-1] == [0]

        # Select s2 and s3
        lst.item(0).setSelected(False)
        lst.item(1).setSelected(True)
        lst.item(2).setSelected(True)
        assert sorted(lst.selected_model_indices()) == [1, 2]

        # Refresh with reordered list -> s2 (now index 2) and s3 (now index 0) stay selected
        lst.set_spectra_names(
            [
                {"fname": "s3", "is_active": True},
                {"fname": "s4", "is_active": True},
                {"fname": "s2", "is_active": True},
            ]
        )
        assert sorted(lst.selected_model_indices()) == [0, 2]
        assert sorted(emissions[-1]) == [0, 2]

    def test_checkbox_toggle_emits_item_changed_and_set_all_checked(self, qapp):
        """Checkbox edits emit itemChanged with fname UserRole+1; set_all_checked updates silently."""
        lst = VSpectraList()
        lst.set_spectra_names(
            [
                {"fname": "spec_0", "is_active": True},
                {"fname": "spec_1", "is_active": True},
            ]
        )
        changed = []
        lst.itemChanged.connect(lambda it: changed.append((it.data(Qt.UserRole + 1), it.checkState())))

        # Toggle via model setData (simulating user click) and via item proxy
        idx0 = lst.model().index(0, 0)
        assert lst.model().setData(idx0, Qt.Unchecked, Qt.CheckStateRole)
        lst.item(1).setCheckState(Qt.Unchecked)

        assert changed == [("spec_0", Qt.Unchecked), ("spec_1", Qt.Unchecked)]

        # set_all_checked updates all rows without emitting per-item itemChanged churn
        changed.clear()
        lst.set_all_checked(True)
        assert changed == []
        assert lst.item(0).checkState() == Qt.Checked
        assert lst.item(1).checkState() == Qt.Checked

    def test_internal_reorder_emits_order_changed(self, qapp):
        """Model reorder moves rows accurately and keeps model indices aligned."""
        lst = VSpectraList()
        lst.set_spectra_names(
            [
                {"fname": "s0", "is_active": True},
                {"fname": "s1", "is_active": True},
                {"fname": "s2", "is_active": True},
                {"fname": "s3", "is_active": True},
            ]
        )
        new_rows = lst.model().reorder_rows([1], 3)
        assert new_rows == [2]
        assert lst.model().fnames == ["s0", "s2", "s1", "s3"]
        assert lst._current_order() == [0, 2, 1, 3]

    def test_large_16384_spectra_population_performance(self, qapp):
        """Populating 16,384 spectra and selecting all stays fast via virtualized model."""
        lst = VSpectraList()
        spectra = [
            {
                "fname": f"spec_{i:05d}_(10.0,20.0)",
                "is_active": True,
                "has_fit": (i % 4 == 0),
                "fit_success": (i % 8 == 0),
                "has_baseline": (i % 4 == 1),
            }
            for i in range(16384)
        ]

        # Warm up once
        lst.set_spectra_names(spectra[:100])

        t0 = time.perf_counter()
        lst.set_spectra_names(spectra)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert lst.count() == 16384
        assert elapsed_ms < 250.0, f"16,384-item list reset took {elapsed_ms:.1f} ms"

        t1 = time.perf_counter()
        lst.select_all()
        selected = lst.selected_model_indices()
        select_ms = (time.perf_counter() - t1) * 1000.0
        assert len(selected) == 16384
        assert selected[0] == 0 and selected[-1] == 16383
        assert select_ms < 150.0, f"select_all + selected_model_indices took {select_ms:.1f} ms"

        # Toggle all checked/unchecked cleanly
        t2 = time.perf_counter()
        lst.set_all_checked(False)
        check_ms = (time.perf_counter() - t2) * 1000.0
        assert check_ms < 100.0, f"set_all_checked took {check_ms:.1f} ms"
        assert lst.item(0).checkState() == Qt.Unchecked
        assert lst.item(16383).checkState() == Qt.Unchecked
