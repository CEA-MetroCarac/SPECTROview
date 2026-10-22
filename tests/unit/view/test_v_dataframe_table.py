import pytest
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QModelIndex, QItemSelectionModel
from PySide6.QtWidgets import QApplication

from spectroview.view.components.v_dataframe_table import DataFrameTableModel, VDataframeTable


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Filename': ['spec_001', 'spec_002', 'spec_003'],
        'X': [10.0, 20.0, 30.0],
        'Y': [1.0, 2.0, 3.0],
        'x0_P1': [520.12, 521.45, 519.88],
        'ampli_P1': [100.5, 200.3, 150.2],
        'fwhm_P1': [3.5, 4.0, 3.8],
        'area_P1': [375.2, 850.1, 605.0],
    })


class TestDataFrameTableModel:
    def test_model_dimensions(self, sample_df, qapp):
        model = DataFrameTableModel(sample_df, fill_colors=True)
        assert model.rowCount() == 3
        assert model.columnCount() == 7

    def test_model_display_data(self, sample_df, qapp):
        model = DataFrameTableModel(sample_df, fill_colors=True)
        idx_00 = model.index(0, 0)
        assert model.data(idx_00, Qt.DisplayRole) == 'spec_001'
        idx_13 = model.index(1, 3)
        assert model.data(idx_13, Qt.DisplayRole) == '521.45'

    def test_model_header_data(self, sample_df, qapp):
        model = DataFrameTableModel(sample_df, fill_colors=True)
        assert model.headerData(0, Qt.Horizontal, Qt.DisplayRole) == 'Filename'
        assert model.headerData(3, Qt.Horizontal, Qt.DisplayRole) == 'x0_P1'
        assert model.headerData(0, Qt.Vertical, Qt.DisplayRole) == '1'
        assert model.headerData(2, Qt.Vertical, Qt.DisplayRole) == '3'

    def test_model_background_colors(self, sample_df, qapp):
        model = DataFrameTableModel(sample_df, fill_colors=True)
        color_x0 = model.data(model.index(0, 3), Qt.BackgroundRole)
        color_ampli = model.data(model.index(0, 4), Qt.BackgroundRole)
        assert color_x0 is not None
        assert color_ampli is not None

    def test_model_sorting(self, sample_df, qapp):
        model = DataFrameTableModel(sample_df, fill_colors=True)
        # Sort by ampli_P1 (col 4) ascending
        model.sort(4, Qt.AscendingOrder)
        assert model.data(model.index(0, 4), Qt.DisplayRole) == '100.5'
        assert model.data(model.index(2, 4), Qt.DisplayRole) == '200.3'

        # Sort by ampli_P1 descending
        model.sort(4, Qt.DescendingOrder)
        assert model.data(model.index(0, 4), Qt.DisplayRole) == '200.3'
        assert model.data(model.index(2, 4), Qt.DisplayRole) == '100.5'

    def test_empty_model(self, qapp):
        model = DataFrameTableModel(pd.DataFrame())
        assert model.rowCount() == 0
        assert model.columnCount() == 0
        assert model.data(model.index(0, 0), Qt.DisplayRole) is None


class TestVDataframeTable:
    def test_show_and_clear(self, sample_df, qapp):
        table = VDataframeTable()
        table.show(sample_df, fill_colors=True)
        assert table.model is not None
        assert table.model.rowCount() == 3
        assert table.model.columnCount() == 7

        table.clear()
        assert table.model.rowCount() == 0

    def test_copy_data(self, sample_df, qapp):
        table = VDataframeTable()
        table.show(sample_df, fill_colors=True)

        # Select a 2x2 range
        sel_model = table.table_view.selectionModel()
        top_left = table.model.index(0, 0)
        bot_right = table.model.index(1, 1)
        sel_model.select(top_left, QItemSelectionModel.Select)
        sel_model.select(bot_right, QItemSelectionModel.Select)

        table._copy_data()
        clipboard_text = QApplication.clipboard().text()
        assert 'spec_001' in clipboard_text
        assert '20.0' in clipboard_text
