"""Unit tests for VMFitModelBuilder."""
import json
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QFileDialog

from spectroview.model.m_settings import MSettings
from spectroview.viewmodel.vm_fit_model_builder import VMFitModelBuilder


@pytest.fixture
def vm_builder(qapp, tmp_path):
    settings = MSettings()
    settings.set_working_folder(str(tmp_path))
    return VMFitModelBuilder(settings)


@pytest.fixture
def sample_model_json(tmp_path):
    model_path = tmp_path / "custom_test_model.json"
    content = {
        "0": {
            "fname": "test_spectrum",
            "range_min": 400.0,
            "range_max": 600.0,
            "peak_labels": ["Peak1"],
            "fit_params": {"method": "Leastsq"},
            "baseline": {"mode": "Linear", "coef": 5},
            "peak_models": {
                "0": {
                    "Lorentzian": {
                        "ampli": {"value": 100.0, "min": 0, "max": 1000},
                        "fwhm": {"value": 5.0, "min": 0, "max": 50},
                        "x0": {"value": 500.0, "min": 450, "max": 550},
                    }
                }
            },
        }
    }
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(content, f)
    return model_path


class TestVMFitModelBuilder:
    def test_load_valid_model_file(self, vm_builder, sample_model_json):
        models_changed_records = []
        model_selected_records = []

        vm_builder.models_changed.connect(models_changed_records.append)
        vm_builder.model_selected.connect(model_selected_records.append)

        success = vm_builder.load_model_file(sample_model_json)
        assert success is True
        assert len(models_changed_records) == 1
        assert "custom_test_model.json" in models_changed_records[0]
        assert model_selected_records == ["custom_test_model.json"]
        assert vm_builder.get_current_model_path() == sample_model_json

    def test_load_same_model_no_duplicates(self, vm_builder, sample_model_json):
        models_changed_records = []
        vm_builder.models_changed.connect(models_changed_records.append)

        vm_builder.load_model_file(sample_model_json)
        vm_builder.load_model_file(sample_model_json)

        assert len(models_changed_records) == 2
        last_models = models_changed_records[-1]
        assert last_models.count("custom_test_model.json") == 1

    def test_load_nonexistent_file(self, vm_builder, tmp_path):
        notifications = []
        vm_builder.notify.connect(notifications.append)

        success = vm_builder.load_model_file(tmp_path / "does_not_exist.json")
        assert success is False
        assert len(notifications) == 1
        assert "Invalid fit model file" in notifications[0]

    def test_load_non_json_file(self, vm_builder, tmp_path):
        txt_path = tmp_path / "model.txt"
        txt_path.write_text("some text")

        notifications = []
        vm_builder.notify.connect(notifications.append)

        success = vm_builder.load_model_file(txt_path)
        assert success is False
        assert len(notifications) == 1
        assert "Invalid fit model file" in notifications[0]

    def test_load_corrupt_json(self, vm_builder, tmp_path):
        bad_json = tmp_path / "corrupt.json"
        bad_json.write_text("{ this is not valid json")

        notifications = []
        vm_builder.notify.connect(notifications.append)

        success = vm_builder.load_model_file(bad_json)
        assert success is False
        assert len(notifications) == 1
        assert "Failed to read fit model" in notifications[0]

    def test_load_json_non_dict(self, vm_builder, tmp_path):
        list_json = tmp_path / "array.json"
        list_json.write_text("[1, 2, 3]")

        notifications = []
        vm_builder.notify.connect(notifications.append)

        success = vm_builder.load_model_file(list_json)
        assert success is False
        assert len(notifications) == 1
        assert "must be a JSON object" in notifications[0]

    def test_pick_and_load_model_dialog(self, vm_builder, sample_model_json):
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(sample_model_json), "JSON files (*.json)")):
            vm_builder.pick_and_load_model()
            assert vm_builder.get_current_model_path() == sample_model_json

    def test_pick_and_load_model_cancelled(self, vm_builder):
        with patch.object(QFileDialog, "getOpenFileName", return_value=("", "")):
            vm_builder.pick_and_load_model()
            assert vm_builder.get_current_model_path() is None
