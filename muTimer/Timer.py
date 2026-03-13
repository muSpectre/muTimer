# MIT License. See LICENSE file for details.

import json
import time
from contextlib import contextmanager

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class Timer:
    """
    Hierarchical timing utility with nested context manager support.

    This class provides fine-grained timing of code sections with support for:
    - Nested timing contexts that track parent-child relationships
    - Accumulation of time across multiple calls to the same timer
    - Call counting for repeated operations
    - Hierarchical summary output in tabular format
    - Optional depth limiting for nested timers

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

    def __init__(self, use_papi=False, track_memory=False):
        """
        Initialize the timer.

        Args:
            use_papi: Ignored (PAPI support has been removed).
            track_memory: If True, tracks process memory (RSS) using psutil.
        """
        self._timers = {}  # name -> {"total": float, "calls": int, "children": list}
        self._stack = []  # stack of (name, start_time) for nesting
        self._roots = []  # top-level timer names in order of first use
        self._track_memory = track_memory

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

    def _collect_rows(self, name, indent=0, parent_time=None, rows=None):
        """Recursively collect timing data as rows for tabular display."""
        if rows is None:
            rows = []

        info = self._timers[name]
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
            }
        )

        # Collect children
        for child in info["children"]:
            self._collect_rows(child, indent + 1, total, rows)

        # Add "other" time if there are children
        if info["children"]:
            children_time = sum(self._timers[c]["total"] for c in info["children"])
            children_memory = sum(self._timers[c]["memory"] for c in info["children"])
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

    def print_summary(self, title=None, name_width=30):
        """
        Print hierarchical timing summary in tabular format.

        Args:
            title: Title for the summary section (default: "Timing Summary")
            name_width: Width of the name column (default: 30)
        """
        if not self._timers:
            print("No timing data collected.")
            return

        # Collect all rows
        all_rows = []
        for root in self._roots:
            self._collect_rows(root, rows=all_rows)

        # Determine title
        if title is None:
            title = "Timing Summary"

        line_width = 92 if self._track_memory else 78
        print(f"\n{'=' * line_width}")
        print(title)
        print(f"{'=' * line_width}")

        if self._track_memory:
            print(
                f"{'Name':<{name_width}} {'Total':>12} {'Calls':>8} "
                f"{'Average':>12} {'% Parent':>10} {'Memory':>12}"
            )
            print(
                f"{'-' * name_width} {'-' * 12} {'-' * 8} {'-' * 12} {'-' * 10} {'-' * 12}"
            )
        else:
            print(
                f"{'Name':<{name_width}} {'Total':>12} {'Calls':>8} "
                f"{'Average':>12} {'% Parent':>10}"
            )
            print(f"{'-' * name_width} {'-' * 12} {'-' * 8} {'-' * 12} {'-' * 10}")

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

            if row["avg"] is not None:
                avg_str = self._format_time(row["avg"])
            else:
                avg_str = f"{'-':>12}"

            if self._track_memory:
                mem_str = self._format_memory(row.get("memory", 0.0))
                print(
                    f"{name:<{name_width}} {total_str} {calls_str} {avg_str} {pct_str} {mem_str}"
                )
            else:
                print(
                    f"{name:<{name_width}} {total_str} {calls_str} {avg_str} {pct_str}"
                )

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
