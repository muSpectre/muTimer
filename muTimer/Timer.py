# MIT License. See LICENSE file for details.

import inspect
import json
import time
from contextlib import contextmanager

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _comm_rank_size(comm):
    """Extract (rank, size) from an mpi4py- or muGrid-style communicator."""
    if hasattr(comm, "Get_rank"):  # mpi4py interface
        return comm.Get_rank(), comm.Get_size()
    return comm.rank, comm.size  # muGrid-style properties


def _gathers_objects(comm):
    """Whether ``comm.gather(obj, root=0)`` gathers Python objects.

    True for mpi4py communicators and for muGrid communicators from 1.6 on,
    whose ``gather`` takes ``root``; muGrid's earlier ``gather`` was an
    array concatenation without ``root`` and is left alone.
    """
    gather = getattr(comm, "gather", None)
    if gather is None:
        return False
    if hasattr(comm, "Get_rank"):  # mpi4py's interface
        return True
    try:
        return "root" in inspect.signature(gather).parameters
    except (TypeError, ValueError):  # no introspectable signature
        return False


def _find_object_gather_comm(comm):
    """Find a communicator capable of gathering Python objects: the given
    one, or the mpi4py communicator a muGrid communicator was made from
    (``mpi4py_comm``); None if there is neither."""
    if _gathers_objects(comm):
        return comm
    candidate = getattr(comm, "mpi4py_comm", None)
    if candidate is not None and _gathers_objects(candidate):
        return candidate
    return None


def _merge_gathered_timers(gathered):
    """Merge per-rank (timers, roots) pairs into a single timer tree.

    For each timer, "total" becomes the mean over all ranks (ranks that
    never entered the timer contribute zero) and "min"/"max" hold the
    extreme per-rank totals. "calls" and "memory" are maxima over ranks.
    """
    nb_ranks = len(gathered)
    merged = {}
    roots = []
    for timers, rank_roots in gathered:
        for name in rank_roots:
            if name not in roots:
                roots.append(name)
        for name, info in timers.items():
            entry = merged.setdefault(
                name,
                {"totals": [], "calls": 0, "children": [], "memory": 0.0},
            )
            entry["totals"].append(info["total"])
            entry["calls"] = max(entry["calls"], info["calls"])
            entry["memory"] = max(entry["memory"], info.get("memory", 0.0))
            for child in info["children"]:
                if child not in entry["children"]:
                    entry["children"].append(child)
    for entry in merged.values():
        totals = entry.pop("totals")
        if len(totals) < nb_ranks:  # timer missing on some ranks
            totals += [0.0] * (nb_ranks - len(totals))
        entry["total"] = sum(totals) / nb_ranks
        entry["min"] = min(totals)
        entry["max"] = max(totals)
    return merged, roots


class Timer:
    """
    Hierarchical timing utility with nested context manager support.

    This class provides fine-grained timing of code sections with support for:
    - Nested timing contexts that track parent-child relationships
    - Accumulation of time across multiple calls to the same timer
    - Call counting for repeated operations
    - Hierarchical summary output in tabular format
    - Optional depth limiting for nested timers
    - MPI-aware summaries: pass a communicator (mpi4py or muGrid style) to
      print a single summary on rank 0 with min/mean/max across ranks

    Note: The `Timer` class is NOT thread-safe. Do not share a single
    `Timer` instance across multiple threads concurrently.

    Example usage:
        from muTimer import Timer

        timer = Timer()

        with timer("outer"):
            # some code
            with timer("inner"):
                # nested code
            with timer("inner"):  # called again - time accumulates
                # more nested code

        timer.print_summary()

    Output:
        ==============================================================================
        Timing Summary
        ==============================================================================
        Name                                  Total    Calls      Average   % Parent
        ------------------------------ ------------ -------- ------------ ----------
        outer                             10.00 ms        1            -          -
          inner                            5.00 ms        2      2.50 ms      50.0%
          (other)                          5.00 ms        -            -      50.0%
        ==============================================================================
    """

    def __init__(self, use_papi=False, track_memory=False, comm=None):
        """
        Initialize the timer.

        Args:
            use_papi: Ignored (PAPI support has been removed).
            track_memory: If True, tracks process memory (RSS) using psutil.
            comm: Optional MPI communicator (mpi4py or muGrid style) used by
                  `print_summary` to produce a single summary on rank 0
                  instead of one per process.
        """
        self._timers = {}  # name -> {"total": float, "calls": int, "children": list}
        self._stack = []  # stack of (name, start_time) for nesting
        self._roots = []  # top-level timer names in order of first use
        self._track_memory = track_memory
        self._comm = comm

        if self._track_memory:
            if not HAS_PSUTIL:
                raise ImportError(
                    "psutil is required for memory tracking. Install with 'pip install psutil'."
                )
            self._process = psutil.Process()

    @contextmanager
    def __call__(self, name: str, max_depth: int = None):
        """
        Context manager for timing a named code section.

        Args:
            name: Identifier for this timing section. Repeated calls with the
                  same name accumulate time and increment the call counter.
            max_depth: Optional maximum depth for nested timing. If the current
                       nesting level exceeds the max_depth of any parent timer,
                       this timer will not record time.

        Yields:
            None

        Example:
            with timer("my_operation"):
                # code to time
                pass
        """
        # Check if depth constraints allow timing
        if not self._is_depth_allowed():
            yield
            return

        # Build hierarchical name based on current stack
        if self._stack:
            parent_name = self._stack[-1][0]
            full_name = f"{parent_name}/{name}"
        else:
            full_name = name

        # Initialize timer if first call
        if full_name not in self._timers:
            self._timers[full_name] = {
                "total": 0.0,
                "calls": 0,
                "children": [],
                "memory": 0.0,
            }
            # Track as root or as child of parent
            if self._stack:
                parent_full = self._stack[-1][0]
                if full_name not in self._timers[parent_full]["children"]:
                    self._timers[parent_full]["children"].append(full_name)
            else:
                if full_name not in self._roots:
                    self._roots.append(full_name)

        # Push onto stack and start timing
        start_mem = self._process.memory_info().rss if self._track_memory else 0.0
        start = time.perf_counter()
        self._stack.append((full_name, start, max_depth, start_mem))

        try:
            yield
        finally:
            # Pop from stack and accumulate time
            elapsed = time.perf_counter() - start
            elapsed_mem = (
                (self._process.memory_info().rss - start_mem)
                if self._track_memory
                else 0.0
            )
            self._stack.pop()
            self._timers[full_name]["total"] += elapsed
            self._timers[full_name]["memory"] += max(0.0, elapsed_mem)
            self._timers[full_name]["calls"] += 1

    def _is_depth_allowed(self):
        """Check parent timers' max_depth constraints.

        Returns:
            bool, True if the timer is within all parent timer max_depth limits
        """
        for i, stack_item in enumerate(self._stack):
            max_depth = stack_item[2]
            if max_depth is not None:
                current_depth = len(self._stack) - i
                if current_depth > max_depth:
                    return False
        return True

    def get_time(self, name):
        """
        Get total accumulated time for a timer.

        Args:
            name: Timer name (can be short name or full hierarchical name)

        Returns:
            Total time in seconds, or 0.0 if timer not found
        """
        # Try exact match first, then search for it in hierarchy
        if name in self._timers:
            return self._timers[name]["total"]
        # Search for timer ending with this name
        matches = []
        for full_name in self._timers:
            if full_name.endswith("/" + name) or full_name == name:
                matches.append(self._timers[full_name]["total"])

        if len(matches) > 1:
            raise ValueError(f"Ambiguous timer name '{name}'. Multiple matches found.")
        elif len(matches) == 1:
            return matches[0]
        return 0.0

    def get_calls(self, name):
        """
        Get call count for a timer.

        Args:
            name: Timer name (can be short name or full hierarchical name)

        Returns:
            Number of calls, or 0 if timer not found
        """
        if name in self._timers:
            return self._timers[name]["calls"]
        matches = []
        for full_name in self._timers:
            if full_name.endswith("/" + name) or full_name == name:
                matches.append(self._timers[full_name]["calls"])

        if len(matches) > 1:
            raise ValueError(f"Ambiguous timer name '{name}'. Multiple matches found.")
        elif len(matches) == 1:
            return matches[0]
        return 0

    def reset(self):
        """Clear all timing data."""
        self._timers.clear()
        self._stack.clear()
        self._roots.clear()

    def _format_time(self, seconds):
        """Format time with appropriate units, fixed width (12 chars)."""
        if seconds < 1e-6:
            return f"{seconds * 1e9:9.2f} ns"
        elif seconds < 1e-3:
            return f"{seconds * 1e6:9.2f} us"
        elif seconds < 1:
            return f"{seconds * 1e3:9.2f} ms"
        elif seconds < 10000:
            return f"{seconds:9.4f} s "
        else:
            return f"{seconds:9.2e} s "

    def _format_memory(self, bytes_val):
        """Format memory with appropriate units."""
        if bytes_val < 1024:
            return f"{bytes_val:9.0f} B "
        elif bytes_val < 1024**2:
            return f"{bytes_val / 1024:9.2f} KB"
        elif bytes_val < 1024**3:
            return f"{bytes_val / 1024**2:9.2f} MB"
        else:
            return f"{bytes_val / 1024**3:9.2f} GB"

    def _collect_rows(self, name, indent=0, parent_time=None, rows=None, timers=None):
        """Recursively collect timing data as rows for tabular display."""
        if rows is None:
            rows = []
        if timers is None:
            timers = self._timers

        info = timers[name]
        total = info["total"]
        calls = info["calls"]
        memory = info["memory"]
        avg = total / calls if calls > 0 else 0

        # Calculate percentage of parent time
        if parent_time and parent_time > 0:
            pct = 100.0 * total / parent_time
        else:
            pct = None

        # Extract short name (last component)
        short_name = name.split("/")[-1]

        # Add row for this timer
        rows.append(
            {
                "indent": indent,
                "name": short_name,
                "total": total,
                "calls": calls,
                "avg": avg if calls > 1 else None,
                "pct": pct,
                "memory": memory,
                "min": info.get("min"),
                "max": info.get("max"),
            }
        )

        # Collect children
        for child in info["children"]:
            self._collect_rows(child, indent + 1, total, rows, timers)

        # Add "other" time if there are children
        if info["children"]:
            children_time = sum(timers[c]["total"] for c in info["children"])
            children_memory = sum(timers[c]["memory"] for c in info["children"])
            other_time = total - children_time
            other_memory = max(0.0, memory - children_memory)
            if other_time < 0:
                import warnings

                warnings.warn(
                    f"Timer overhead detected or total time is less than children time for '{name}'."
                )
                other_time = 0.0
            if other_time > 1e-9 or other_memory > 0:  # Only show if meaningful
                other_pct = 100.0 * other_time / total if total > 0 else 0
                rows.append(
                    {
                        "indent": indent + 1,
                        "name": "(other)",
                        "total": other_time,
                        "calls": None,
                        "avg": None,
                        "pct": other_pct,
                        "memory": other_memory,
                    }
                )

        return rows

    def print_summary(self, title=None, name_width=30, comm=None):
        """
        Print hierarchical timing summary in tabular format.

        When running under MPI on more than one process and a communicator
        is available (either passed here or to the constructor), timing data
        is gathered to rank 0 and a single summary is printed there, showing
        the mean, minimum and maximum time across ranks; all other ranks
        print nothing. Gathering needs a communicator whose ``gather`` takes
        Python objects -- mpi4py's, or muGrid's from 1.6 on; without one,
        rank 0 prints its local timings only.

        Args:
            title: Title for the summary section (default: "Timing Summary")
            name_width: Width of the name column (default: 30)
            comm: Optional MPI communicator (mpi4py or muGrid style).
                  Overrides the communicator passed to the constructor.
        """
        if comm is None:
            comm = self._comm
        rank, size = (0, 1) if comm is None else _comm_rank_size(comm)

        timers = self._timers
        roots = self._roots
        parallel = False
        if size > 1:
            gather_comm = _find_object_gather_comm(comm)
            if gather_comm is not None:
                # Collective call: every rank must participate in the gather
                gathered = gather_comm.gather((self._timers, self._roots), root=0)
                if rank == 0:
                    timers, roots = _merge_gathered_timers(gathered)
                    parallel = True
            if rank != 0:
                return

        if not timers:
            print("No timing data collected.")
            return

        # Collect all rows
        all_rows = []
        for root in roots:
            self._collect_rows(root, rows=all_rows, timers=timers)

        # Determine title
        if title is None:
            title = "Timing Summary"
        if parallel:
            title += f" ({size} MPI processes)"

        # In parallel mode, the Total/Average columns are replaced by
        # Mean/Min/Max across ranks (13 extra characters)
        line_width = 92 if self._track_memory else 78
        if parallel:
            line_width += 13
        print(f"\n{'=' * line_width}")
        print(title)
        print(f"{'=' * line_width}")

        if parallel:
            time_header = f"{'Mean':>12} {'Min':>12} {'Max':>12}"
            time_rule = f"{'-' * 12} {'-' * 12} {'-' * 12}"
        else:
            time_header = f"{'Total':>12} {'Calls':>8} {'Average':>12}"
            time_rule = f"{'-' * 12} {'-' * 8} {'-' * 12}"
        if parallel:
            time_header += f" {'Calls':>8}"
            time_rule += f" {'-' * 8}"
        if self._track_memory:
            print(
                f"{'Name':<{name_width}} {time_header} {'% Parent':>10} {'Memory':>12}"
            )
            print(f"{'-' * name_width} {time_rule} {'-' * 10} {'-' * 12}")
        else:
            print(f"{'Name':<{name_width}} {time_header} {'% Parent':>10}")
            print(f"{'-' * name_width} {time_rule} {'-' * 10}")

        # Print rows
        for row in all_rows:
            # Build indented name
            prefix = "  " * row["indent"]
            name = f"{prefix}{row['name']}"
            if len(name) > name_width:
                name = name[: name_width - 3] + "..."

            # Format columns
            total_str = self._format_time(row["total"])

            if row["calls"] is not None:
                calls_str = f"{row['calls']:>8}"
            else:
                calls_str = f"{'-':>8}"

            if row["pct"] is not None:
                pct_str = f"{row['pct']:>9.1f}%"
            else:
                pct_str = f"{'-':>10}"

            if parallel:
                if row.get("min") is not None:
                    min_str = self._format_time(row["min"])
                    max_str = self._format_time(row["max"])
                else:
                    min_str = f"{'-':>12}"
                    max_str = f"{'-':>12}"
                time_cols = f"{total_str} {min_str} {max_str} {calls_str}"
            else:
                if row["avg"] is not None:
                    avg_str = self._format_time(row["avg"])
                else:
                    avg_str = f"{'-':>12}"
                time_cols = f"{total_str} {calls_str} {avg_str}"

            if self._track_memory:
                mem_str = self._format_memory(row.get("memory", 0.0))
                print(f"{name:<{name_width}} {time_cols} {pct_str} {mem_str}")
            else:
                print(f"{name:<{name_width}} {time_cols} {pct_str}")

        print(f"{'=' * line_width}")

    def _build_tree(self, name):
        """Recursively build a tree structure for a timer and its children."""
        info = self._timers[name]
        total = info["total"]
        calls = info["calls"]

        result = {
            "name": name.split("/")[-1],
            "total_seconds": total,
            "calls": calls,
            "avg_seconds": total / calls if calls > 0 else 0,
        }

        if self._track_memory:
            result["memory_bytes"] = info.get("memory", 0.0)

        if info["children"]:
            result["children"] = [self._build_tree(child) for child in info["children"]]
            children_time = sum(self._timers[c]["total"] for c in info["children"])
            other_time = total - children_time
            if other_time > 1e-9:
                result["other_seconds"] = other_time

            if self._track_memory:
                children_memory = sum(
                    self._timers[c]["memory"] for c in info["children"]
                )
                other_memory = max(0.0, info.get("memory", 0.0) - children_memory)
                if other_memory > 0:
                    result["other_memory_bytes"] = other_memory

        return result

    def to_dict(self):
        """
        Return timing data as a hierarchical dictionary.

        Returns:
            Dictionary with hierarchical timer structure suitable for JSON export.
        """
        return {"timers": [self._build_tree(root) for root in self._roots]}

    def to_json(self, indent=2):
        """
        Return timing data as a JSON string.

        Args:
            indent: Indentation level for pretty printing (default: 2)

        Returns:
            JSON string representation of timing data.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def summary_dict(self):
        """
        Return timing data as a flat dictionary for programmatic access.

        Returns:
            Dictionary with timer names as keys and dicts containing
            'total', 'calls', and 'children' as values.
        """
        result = {}
        for name, info in self._timers.items():
            entry = {
                "total": info["total"],
                "calls": info["calls"],
                "avg": info["total"] / info["calls"] if info["calls"] > 0 else 0,
                "children": list(info["children"]),
            }
            if self._track_memory:
                entry["memory"] = info.get("memory", 0.0)
            result[name] = entry
        return result
