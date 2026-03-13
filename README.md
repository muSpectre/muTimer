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

## License
muTimer is free software, distributed under the GNU Lesser General Public License version 3 or later.
