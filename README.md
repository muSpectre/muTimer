# muTimer

muTimer is a hierarchical timing utility with nested context manager support. It provides fine-grained timing of code sections.

## Features
- Nested timing contexts that track parent-child relationships
- Accumulation of time across multiple calls to the same timer
- Call counting for repeated operations
- Hierarchical summary output in tabular format
- Optional depth limiting for nested timers

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
