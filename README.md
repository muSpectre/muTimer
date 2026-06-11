# muTimer

muTimer is a hierarchical timing utility with nested context manager support. It provides fine-grained timing of code sections.

## Features
- Nested timing contexts that track parent-child relationships
- Accumulation of time across multiple calls to the same timer
- Call counting for repeated operations
- Hierarchical summary output in tabular format
- Optional depth limiting for nested timers
- MPI-aware summaries (single output on rank 0 with min/mean/max across ranks)

## Usage
```python
from muTimer import Timer

timer = Timer()

with timer("outer"):
    # some code
    with timer("inner"):
        # nested code
    with timer("inner"):  # called again - time accumulates
        # more nested code

timer.print_summary()
```

Output:

```text
==============================================================================
Timing Summary
==============================================================================
Name                                  Total    Calls      Average   % Parent
------------------------------ ------------ -------- ------------ ----------
outer                              22.55 ms        1            -          -
  inner                            12.50 ms        2      6.25 ms      55.4%
  (other)                          10.06 ms        -            -      44.6%
==============================================================================
```

### MPI Support
When running under MPI, pass a communicator to avoid every rank printing its
own summary. Both mpi4py communicators and muGrid communicators are accepted.

```python
from mpi4py import MPI
from muTimer import Timer

timer = Timer(comm=MPI.COMM_WORLD)

with timer("outer"):
    # some code
    pass

timer.print_summary()  # printed once, on rank 0
```

Timing data is gathered to rank 0, which prints a single summary showing the
mean, minimum and maximum time spent in each section across all ranks:

```text
===========================================================================================
Timing Summary (4 MPI processes)
===========================================================================================
Name                                   Mean          Min          Max    Calls   % Parent
------------------------------ ------------ ------------ ------------ -------- ----------
outer                              23.30 ms     22.55 ms     24.05 ms        1          -
  inner                            12.95 ms     12.50 ms     13.40 ms        2      55.6%
  (other)                          10.35 ms            -            -        -      44.4%
===========================================================================================
```

If the communicator cannot gather Python objects (no underlying mpi4py
communicator), the summary is still printed only on rank 0, using that rank's
local timings.

muTimer has no dependency on mpi4py (or any other MPI package): the
communicator is duck-typed. Anything that provides the mpi4py communicator
interface works, e.g. [NuMPI](https://github.com/IMTEK-Simulation/NuMPI)'s
`MPI` module, which falls back to a serial stub when mpi4py is not installed:

```python
from NuMPI import MPI  # mpi4py if installed, serial stub otherwise
from muTimer import Timer

timer = Timer(comm=MPI.COMM_WORLD)
```

### Memory Tracking
You can also track memory usage (Resident Set Size) by enabling `track_memory=True`. This requires the `psutil` package.

```python
import time
from muTimer import Timer

# Create a timer with memory tracking enabled
timer = Timer(track_memory=True)

with timer("outer"):
    # allocate some memory
    large_list = [0] * 1000000
    time.sleep(0.01)
    
    with timer("inner"):
        another_list = [1] * 2000000
        time.sleep(0.01)

    with timer("inner"):
        more_memory = [2] * 500000
        time.sleep(0.005)

timer.print_summary()
```

Output:

```text
============================================================================================
Timing Summary
============================================================================================
Name                                  Total    Calls      Average   % Parent       Memory
------------------------------ ------------ -------- ------------ ---------- ------------
outer                              33.74 ms        1            -          -     26.78 MB
  inner                            20.52 ms        2     10.26 ms      60.8%     19.12 MB
  (other)                          13.23 ms        -            -      39.2%      7.66 MB
============================================================================================
```

## License
muTimer is distributed under the MIT License.
