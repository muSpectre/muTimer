import time

import pytest

from muTimer import Timer


def test_basic_timing():
    timer = Timer()
    with timer("test_basic"):
        time.sleep(0.01)

    assert timer.get_calls("test_basic") == 1
    assert timer.get_time("test_basic") >= 0.01


def test_accumulation():
    timer = Timer()
    for _ in range(3):
        with timer("test_accum"):
            time.sleep(0.01)

    assert timer.get_calls("test_accum") == 3
    assert timer.get_time("test_accum") >= 0.03


def test_hierarchy_tracking():
    timer = Timer()
    with timer("outer"):
        time.sleep(0.01)
        with timer("inner"):
            time.sleep(0.01)

    assert timer.get_calls("outer") == 1
    assert timer.get_calls("inner") == 1
    assert timer.get_calls("outer/inner") == 1

    assert timer.get_time("outer") >= timer.get_time("inner")
    assert "outer" in timer._roots
    assert "inner" not in timer._roots


def test_max_depth():
    timer = Timer()
    with timer("depth0", max_depth=0):
        with timer("depth1"):
            pass

    assert timer.get_calls("depth0") == 1
    assert timer.get_calls("depth1") == 0


def test_exceptions_in_context():
    timer = Timer()
    try:
        with timer("fail_block"):
            time.sleep(0.01)
            raise ValueError("Test error")
    except ValueError:
        pass

    assert timer.get_calls("fail_block") == 1
    assert timer.get_time("fail_block") >= 0.01
    assert len(timer._stack) == 0


def test_zero_time_operations():
    timer = Timer()
    with timer("zero"):
        pass
    assert timer.get_calls("zero") == 1
    assert timer.get_time("zero") >= 0.0


def test_ambiguous_names():
    timer = Timer()
    with timer("outer1"):
        with timer("inner"):
            pass
    with timer("outer2"):
        with timer("inner"):
            pass

    with pytest.raises(ValueError, match="Ambiguous timer name 'inner'"):
        timer.get_time("inner")

    with pytest.raises(ValueError, match="Ambiguous timer name 'inner'"):
        timer.get_calls("inner")

    assert timer.get_time("outer1/inner") >= 0.0
    assert timer.get_calls("outer2/inner") == 1


def test_negative_other_time(capsys):
    timer = Timer()
    with timer("outer"):
        with timer("inner"):
            pass

    # Manually hack times to force negative other_time
    timer._timers["outer"]["total"] = 1.0
    timer._timers["outer/inner"]["total"] = 2.0

    with pytest.warns(UserWarning, match="Timer overhead detected"):
        timer.print_summary()


def test_format_time():
    timer = Timer()
    assert timer._format_time(1e-7).strip().endswith("ns")
    assert timer._format_time(1e-4).strip().endswith("us")
    assert timer._format_time(1e-1).strip().endswith("ms")
    assert timer._format_time(1.0).strip().endswith("s")
    assert timer._format_time(10000.0).strip().endswith("s")

    # Test length for large value
    large_format = timer._format_time(1e5)
    assert len(large_format) <= 12


def test_export():
    timer = Timer()
    with timer("outer"):
        with timer("inner"):
            pass

    d = timer.to_dict()
    assert "timers" in d
    assert d["timers"][0]["name"] == "outer"
    assert d["timers"][0]["children"][0]["name"] == "inner"

    j = timer.to_json()
    assert "outer" in j
    assert "inner" in j


def test_memory_tracking():
    timer = Timer(track_memory=True)
    with timer("mem_test"):
        # Allocate some memory to ensure we exercise the code path
        _ = [0] * 10000

    assert timer.get_calls("mem_test") == 1
    # Check that memory is non-negative and present
    d = timer.summary_dict()
    assert "memory" in d["mem_test"]
    assert d["mem_test"]["memory"] >= 0.0


def test_format_memory():
    timer = Timer()
    assert " B" in timer._format_memory(500)
    assert "KB" in timer._format_memory(2048)
    assert "MB" in timer._format_memory(2 * 1024**2)
    assert "GB" in timer._format_memory(2 * 1024**3)
