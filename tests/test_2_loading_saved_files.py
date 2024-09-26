import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from examples.ex_loading_saved_files import ex_loading_saved_files

def test_2_loading_saved_files(qtbot):
    ex_loading_saved_files(run_app=False) 
