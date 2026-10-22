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


class FakeMPIComm:
    """Minimal mpi4py-like communicator backed by a shared store."""

    def __init__(self, rank, size, store):
        self._rank = rank
        self._size = size
        self._store = store

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def gather(self, obj, root=0):
        self._store[self._rank] = obj
        if self._rank == root:
            return [self._store[r] for r in range(self._size)]
        return None


class FakeMuGridComm:
    """Minimal muGrid-style communicator wrapping an mpi4py communicator."""

    def __init__(self, mpi4py_comm):
        self.mpi4py_comm = mpi4py_comm

    @property
    def rank(self):
        return self.mpi4py_comm.Get_rank()

    @property
    def size(self):
        return self.mpi4py_comm.Get_size()


class FakeNoGatherComm:
    """muGrid-style communicator without an underlying mpi4py communicator."""

    def __init__(self, rank, size):
        self.rank = rank
        self.size = size


def _make_rank_timer(comm, seconds):
    timer = Timer(comm=comm)
    timer._timers["work"] = {
        "total": seconds,
        "calls": 2,
        "children": [],
        "memory": 0.0,
    }
    timer._roots.append("work")
    return timer


def test_mpi_summary_prints_once(capsys):
    store = {}
    comm0 = FakeMPIComm(0, 2, store)
    comm1 = FakeMPIComm(1, 2, store)
    timer0 = _make_rank_timer(comm0, 1.0)
    timer1 = _make_rank_timer(comm1, 3.0)

    # Non-root ranks contribute their data but print nothing
    timer1.print_summary()
    assert capsys.readouterr().out == ""

    timer0.print_summary()
    out = capsys.readouterr().out
    assert "Timing Summary (2 MPI processes)" in out
    assert "Mean" in out and "Min" in out and "Max" in out
    # mean 2 s, min 1 s, max 3 s
    assert "2.0000 s" in out
    assert "1.0000 s" in out
    assert "3.0000 s" in out


def test_mpi_summary_with_mugrid_style_comm(capsys):
    store = {}
    comm0 = FakeMuGridComm(FakeMPIComm(0, 2, store))
    comm1 = FakeMuGridComm(FakeMPIComm(1, 2, store))
    timer0 = _make_rank_timer(comm0, 1.0)
    timer1 = _make_rank_timer(comm1, 3.0)

    timer1.print_summary()
    assert capsys.readouterr().out == ""

    timer0.print_summary()
    out = capsys.readouterr().out
    assert "Timing Summary (2 MPI processes)" in out
    assert "2.0000 s" in out


def test_mpi_summary_timer_missing_on_some_ranks(capsys):
    store = {}
    comm0 = FakeMPIComm(0, 2, store)
    comm1 = FakeMPIComm(1, 2, store)
    timer0 = _make_rank_timer(comm0, 4.0)
    timer1 = Timer(comm=comm1)  # never timed anything

    timer1.print_summary()
    assert capsys.readouterr().out == ""

    timer0.print_summary()
    out = capsys.readouterr().out
    # mean 2 s, min 0 s, max 4 s
    assert "2.0000 s" in out
    assert "0.00 ns" in out
    assert "4.0000 s" in out


def test_mpi_summary_without_gather_prints_rank0_only(capsys):
    timer1 = _make_rank_timer(FakeNoGatherComm(1, 2), 3.0)
    timer1.print_summary()
    assert capsys.readouterr().out == ""

    timer0 = _make_rank_timer(FakeNoGatherComm(0, 2), 1.0)
    timer0.print_summary()
    out = capsys.readouterr().out
    # Falls back to the local (serial-style) summary on rank 0
    assert "Timing Summary" in out
    assert "MPI processes" not in out
    assert "1.0000 s" in out


def test_comm_argument_overrides_constructor(capsys):
    timer = _make_rank_timer(None, 1.0)
    timer.print_summary(comm=FakeNoGatherComm(1, 2))
    assert capsys.readouterr().out == ""


def test_serial_comm_keeps_serial_format(capsys):
    timer = _make_rank_timer(FakeNoGatherComm(0, 1), 1.0)
    timer.print_summary()
    out = capsys.readouterr().out
    assert "Total" in out and "Average" in out
    assert "MPI processes" not in out


def test_serial_mpi4py_style_comm_keeps_serial_format(capsys):
    # Shape of mpi4py's COMM_WORLD and of NuMPI's serial stub
    timer = _make_rank_timer(FakeMPIComm(0, 1, {}), 1.0)
    timer.print_summary()
    out = capsys.readouterr().out
    assert "Total" in out and "Average" in out
    assert "MPI processes" not in out


def test_object_gather_detection():
    from muTimer.Timer import _find_object_gather_comm

    class Mpi4pyLike:
        def Get_rank(self):
            return 0

        def gather(self, obj, root=0):
            return [obj]

    class MuGridLike:  # muGrid >= 1.6: gather takes root, no Get_rank
        rank = 0

        def gather(self, obj, root=0):
            return [obj]

    class OldMuGridLike:  # muGrid < 1.6: gather concatenated arrays
        rank = 0
        mpi4py_comm = None

        def gather(self, arg):
            return arg

    class Wrapped(OldMuGridLike):
        mpi4py_comm = Mpi4pyLike()

    assert _find_object_gather_comm(Mpi4pyLike()) is not None
    assert _find_object_gather_comm(MuGridLike()) is not None
    assert _find_object_gather_comm(OldMuGridLike()) is None
    assert isinstance(_find_object_gather_comm(Wrapped()), Mpi4pyLike)
    assert _find_object_gather_comm(object()) is None
