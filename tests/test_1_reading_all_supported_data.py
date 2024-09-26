# # Test_1
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import pytest
# # from examples.ex_reading_all_supported_data import ex_reading_all_supported_data

# def test_reading_all_supported_data(qtbot):
#     print("test")
#     # ex_reading_all_supported_data(run_app=False) 

import pytest
from PySide6.QtWidgets import QApplication

@pytest.fixture(scope="module", autouse=True)
def app(qtbot):
    # This is how you create a QApplication instance for testing
    app = QApplication([])

    # Ensure QApplication is quit after tests
    yield app
    app.quit()

def test_some_functionality(qtbot):
    print("test")
    assert True  # Replace with your actual test
