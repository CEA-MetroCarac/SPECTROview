import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from examples.ex_reading_all_supported_data import ex_reading_all_supported_data

def test_1_reading_all_supported_data(qtbot):
    ex_reading_all_supported_data(run_app=False) 
