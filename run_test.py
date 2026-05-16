import sys
import pytest
sys.exit(pytest.main(["-v", "-s", "tests/test1_all_workflows.py::TestMapsWorkflow::test_process_single_map"]))
