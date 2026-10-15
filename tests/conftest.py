"""
Registers the test tags and starts the 3-rank re-runs of test_parallel.py when the session starts, so
they run beside the serial tests instead of after them, and moves the tests that wait for them to the end.

Every test carries one of two tags (select with -m, e.g. pytest -m verification):
  verification  the test checks derivatives or output values against an analytic, finite-difference or
                reference result
  unit_test     any other test: bookkeeping, caching, validation, file layout, API behaviour
test_parallel.py, which re-runs the other modules under MPI, carries both; the marker expression given
to pytest is passed on to those runs.
"""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "verification: checks derivatives or output values against an "
                                       "analytic, finite-difference or reference result")
    config.addinivalue_line("markers", "unit_test: any other test (bookkeeping, caching, validation, "
                                       "file layout, API behaviour)")


@pytest.hookimpl(trylast=True)       # after -k / -m have deselected
def pytest_collection_modifyitems(config, items):
    waiting = [i for i in items if "test_parallel.py" in i.nodeid]
    if not waiting:
        return
    items[:] = [i for i in items if i not in waiting] + waiting
    if os.environ.get("DRAGONFLY_PARALLEL_TEST_CHILD") != "1" and not config.option.collectonly:
        import test_parallel
        test_parallel.start_launches(config.getoption("markexpr"))
