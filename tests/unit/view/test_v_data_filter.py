"""Unit tests for VDataFilter component."""

import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from spectroview.view.components.v_data_filter import VDataFilter


@pytest.fixture
def filter_widget(qapp):
    widget = VDataFilter()
    widget.show()
    return widget


class TestVDataFilterAdd:
    def test_add_filter_keeps_expression_in_query(self, filter_widget):
        """Requirement 5: QLineEdit expression should remain after adding filter."""
        filter_widget.filter_query.setText("Intensity > 100")
        filter_widget._on_add_filter()

        assert filter_widget.filter_listbox.count() == 1
        assert filter_widget.filter_query.text() == "Intensity > 100"

        # Verify added filter state
        filters = filter_widget.get_filters()
        assert len(filters) == 1
        assert filters[0]["expression"] == "Intensity > 100"
        assert filters[0]["state"] is True

    def test_sequential_add_by_modifying_query(self, filter_widget):
        """User can edit expression and press Enter repeatedly to add multiple filters."""
        filter_widget.filter_query.setText("Intensity > 100")
        filter_widget._on_add_filter()

        filter_widget.filter_query.setText("Intensity > 200")
        filter_widget._on_add_filter()

        filter_widget.filter_query.setText("Intensity > 300")
        filter_widget._on_add_filter()

        assert filter_widget.filter_listbox.count() == 3
        expressions = [f["expression"] for f in filter_widget.get_filters()]
        assert expressions == ["Intensity > 100", "Intensity > 200", "Intensity > 300"]
        assert filter_widget.filter_query.text() == "Intensity > 300"

    def test_add_duplicate_activates_without_duplicate_row(self, filter_widget):
        """Adding an existing filter activates it and avoids duplicate rows."""
        filter_widget.add_filter("Intensity > 100", state=False)
        assert filter_widget.get_filters()[0]["state"] is False

        filter_widget.filter_query.setText("Intensity > 100")
        filter_widget._on_add_filter()

        assert filter_widget.filter_listbox.count() == 1
        assert filter_widget.get_filters()[0]["state"] is True


class TestVDataFilterSelectionAndHover:
    def test_hover_does_not_populate_query(self, filter_widget):
        """Requirement 4: Hovering over filter item or checkbox does NOT populate QLineEdit."""
        filter_widget.add_filter("Intensity > 100")
        filter_widget.add_filter("FWHM < 5")
        filter_widget.filter_query.clear()

        item0 = filter_widget.filter_listbox.item(0)
        cb0 = filter_widget.filter_listbox.itemWidget(item0)
        item1 = filter_widget.filter_listbox.item(1)
        cb1 = filter_widget.filter_listbox.itemWidget(item1)

        # Mouse move / hover over item 0
        QTest.mouseMove(cb0)
        assert filter_widget.filter_query.text() == ""

        # Mouse move / hover over item 1
        QTest.mouseMove(cb1)
        assert filter_widget.filter_query.text() == ""

        # Mouse move on viewport
        QTest.mouseMove(filter_widget.filter_listbox.viewport(), filter_widget.filter_listbox.visualItemRect(item0).center())
        assert filter_widget.filter_query.text() == ""

    def test_click_checkbox_populates_query_and_toggles(self, filter_widget, qtbot):
        """Requirement 4: Clicking checkbox explicitly loads expression into QLineEdit."""
        filter_widget.add_filter("Intensity > 100", state=False)
        filter_widget.filter_query.clear()

        item = filter_widget.filter_listbox.item(0)
        cb = filter_widget.filter_listbox.itemWidget(item)
        assert cb.isChecked() is False

        # Click the checkbox indicator
        qtbot.mouseClick(cb, Qt.LeftButton, pos=QPoint(10, 10))
        assert filter_widget.filter_query.text() == "Intensity > 100"
        assert cb.isChecked() is True

    def test_click_item_row_populates_query(self, filter_widget):
        """Requirement 4: Clicking item row explicitly loads expression into QLineEdit."""
        filter_widget.add_filter("Peak position > 500")
        filter_widget.filter_query.clear()

        item = filter_widget.filter_listbox.item(0)
        rect = filter_widget.filter_listbox.visualItemRect(item)
        QTest.mouseClick(filter_widget.filter_listbox.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())

        assert filter_widget.filter_query.text() == "Peak position > 500"

    def test_keyboard_navigation_populates_query(self, filter_widget):
        """Keyboard navigation (Up/Down) updates query on selection change."""
        filter_widget.add_filter("Filter 1")
        filter_widget.add_filter("Filter 2")
        filter_widget.filter_query.clear()

        filter_widget.filter_listbox.setFocus()
        item0 = filter_widget.filter_listbox.item(0)
        filter_widget.filter_listbox.setCurrentItem(item0)

        QTest.keyClick(filter_widget.filter_listbox, Qt.Key_Down)
        assert filter_widget.filter_query.text() == "Filter 2"

        QTest.keyClick(filter_widget.filter_listbox, Qt.Key_Up)
        assert filter_widget.filter_query.text() == "Filter 1"


class TestVDataFilterGraphSwitching:
    def test_switching_graphs_updates_states_without_removing_filters(self, filter_widget):
        """Requirements 2 & 3: Switching graphs preserves available filter list and updates checked states."""
        # Session has 3 filters
        filter_widget.add_filter("Intensity > 100", state=True)
        filter_widget.add_filter("FWHM < 5", state=False)
        filter_widget.add_filter("Peak position > 500", state=True)

        assert filter_widget.filter_listbox.count() == 3

        # Switch to Graph B which only has FWHM < 5 active
        graph_b_filters = [{"expression": "FWHM < 5", "state": True}]
        filter_widget.set_filters(graph_b_filters)

        # Count should still be 3 -- no filters deleted!
        assert filter_widget.filter_listbox.count() == 3

        states = {f["expression"]: f["state"] for f in filter_widget.get_filters()}
        assert states == {
            "Intensity > 100": False,
            "FWHM < 5": True,
            "Peak position > 500": False,
        }

    def test_switching_to_graph_with_new_filter_appends_definition(self, filter_widget):
        """If a graph has a filter expression not yet in available list, it is added."""
        filter_widget.add_filter("Intensity > 100", state=True)

        # Graph C has a slot filter
        graph_c_filters = [
            {"expression": "Intensity > 100", "state": True},
            {"expression": "Slot == 2", "state": True},
        ]
        filter_widget.update_filter_states(graph_c_filters)

        assert filter_widget.filter_listbox.count() == 2
        states = {f["expression"]: f["state"] for f in filter_widget.get_filters()}
        assert states == {
            "Intensity > 100": True,
            "Slot == 2": True,
        }

    def test_switching_to_empty_graph_unchecks_all_without_deleting(self, filter_widget):
        """Switching to graph with no filters leaves available filters intact but unchecked."""
        filter_widget.add_filter("Intensity > 100", state=True)
        filter_widget.add_filter("Area > 1000", state=True)

        filter_widget.set_filters([])

        assert filter_widget.filter_listbox.count() == 2
        states = {f["expression"]: f["state"] for f in filter_widget.get_filters()}
        assert states == {
            "Intensity > 100": False,
            "Area > 1000": False,
        }

    def test_switching_graphs_does_not_clear_user_entered_query(self, filter_widget):
        """Switching graphs should not overwrite whatever expression user is currently drafting."""
        filter_widget.add_filter("Intensity > 100", state=True)
        filter_widget.filter_query.setText("Custom drafting expression...")

        filter_widget.set_filters([{"expression": "Intensity > 100", "state": False}])

        assert filter_widget.filter_query.text() == "Custom drafting expression..."


class TestVDataFilterRemovalAndClear:
    def test_remove_filter(self, filter_widget):
        """User can explicitly remove a filter definition."""
        filter_widget.add_filter("Filter 1")
        filter_widget.add_filter("Filter 2")

        item1 = filter_widget.filter_listbox.item(0)
        filter_widget.filter_listbox.setCurrentItem(item1)
        item1.setSelected(True)

        filter_widget._on_remove_filter()

        assert filter_widget.filter_listbox.count() == 1
        assert filter_widget.get_filters()[0]["expression"] == "Filter 2"

    def test_clear_filters(self, filter_widget):
        """clear_filters removes all items and clears query."""
        filter_widget.add_filter("Filter 1")
        filter_widget.clear_filters()

        assert filter_widget.filter_listbox.count() == 0
        assert filter_widget.filter_query.text() == ""
